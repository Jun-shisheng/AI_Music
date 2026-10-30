"""Audit English phone-frame capacity and apply explicit, reversible SVS lyric-boundary recipes.

Never changes pitches, rests, total duration, dictionary pronunciations, or a singer profile.
The chorus_v3 recipe addresses three observed rest-crossing word assignments only.
"""
from __future__ import annotations

import argparse
import copy
from decimal import Decimal
import json
from pathlib import Path
import sys

import numpy as np

from . import audio
from .workflow import validate_metadata

FIELDS=("duration","text","phoneme","note_pitch","note_type")


def unpack(meta:dict)->list[dict]:
    arrays={field:meta[field].split() for field in FIELDS}
    if len({len(value) for value in arrays.values()})!=1:
        raise ValueError("Aligned metadata arrays have different lengths")
    events=[];cursor=Decimal(0)
    for index in range(len(arrays["duration"])):
        duration=Decimal(arrays["duration"][index])
        if duration<=0:
            raise ValueError("Metadata durations must be positive")
        events.append(dict(start=cursor,end=cursor+duration,text=arrays["text"][index],phoneme=arrays["phoneme"][index],
            pitch=int(arrays["note_pitch"][index]),kind=int(arrays["note_type"][index])))
        cursor+=duration
    return events


def onset_words(events:list[dict])->list[str]:
    return [e["text"] for e in events if e["kind"]==2]


def pack(meta:dict,events:list[dict])->dict:
    result=copy.deepcopy(meta)
    result.update(duration=" ".join(format(e["end"]-e["start"],".6f") for e in events),
        text=" ".join(e["text"] for e in events),phoneme=" ".join(e["phoneme"] for e in events),
        note_pitch=" ".join(str(e["pitch"]) for e in events),note_type=" ".join(str(e["kind"]) for e in events))
    return result


def capacity(events:list[dict])->dict:
    rows=[];crossings=[]
    for index,event in enumerate(events):
        start=int(np.round(float(event["start"])*50));end=int(np.round(float(event["end"])*50))
        english=event["phoneme"].startswith("en_")
        phones=event["phoneme"][3:].split("-") if english else [event["phoneme"]]
        width=len(phones)+1 if english else 1
        cursor=start+1;body=0;cycles=0
        while cursor+width<end:
            body+=width;cycles+=1;cursor+=width
        rows.append(dict(index=index,text=event["text"],kind=event["kind"],pitch=event["pitch"],
            start_seconds=float(event["start"]),end_seconds=float(event["end"]),slot_frames=end-start,
            phone_count=len(phones) if english else 0,phone_body_frames=body if english else 0,
            complete_phone_passes=cycles if english else 0,predicted_pad_frames=max(0,end-start-body-2)))
        if 0<index<len(events)-1 and event["kind"]==1:
            previous,following=events[index-1],events[index+1]
            if previous["text"]==following["text"] and following["kind"]==3:
                crossings.append(dict(word=previous["text"],rest_start_seconds=float(event["start"]),
                    rest_end_seconds=float(event["end"]),following_index=index+1))
    return dict(note_slots=rows,rest_crossing_word_continuations=crossings,
        zero_body_english_slots=[r for r in rows if r["phone_count"] and not r["phone_body_frames"]],
        zero_body_new_word_onsets=[r for r in rows if r["phone_count"] and r["kind"]==2 and not r["phone_body_frames"]])


def merge_continuations(events:list[dict])->list[dict]:
    merged=[]
    for event in events:
        if merged and event["kind"]==3 and all(event[k]==merged[-1][k] for k in ["text","phoneme","pitch"]) and event["start"]==merged[-1]["end"]:
            merged[-1]["end"]=event["end"]
        else:
            merged.append(event.copy())
    return merged


def pitch_invariants(before:list[dict],after:list[dict])->dict:
    if before[-1]["end"]!=after[-1]["end"] or onset_words(before)!=onset_words(after):
        raise ValueError("Patch changed total duration or new-word order")
    boundaries=sorted({x for event in before+after for x in [event["start"],event["end"]]})
    for start,end in zip(boundaries,boundaries[1:]):
        middle=(start+end)/2
        a=next(e for e in before if e["start"]<=middle<e["end"])
        b=next(e for e in after if e["start"]<=middle<e["end"])
        if a["pitch"]!=b["pitch"] or (a["kind"]==1)!=(b["kind"]==1):
            raise ValueError("Patch changed a pitch or rest interval")
    pronunciations=lambda events:{e["text"]:e["phoneme"] for e in events if e["kind"]!=1}
    if pronunciations(before)!=pronunciations(after):
        raise ValueError("Patch changed a dictionary pronunciation")
    return dict(total_duration_decimal_seconds=str(before[-1]["end"]),pitch_timeline_identical=True,
        rest_timeline_identical=True,new_word_order_identical=True,new_word_spelling_identical=True,
        dictionary_pronunciations_identical=True,new_word_sequence=onset_words(after))


def chorus_v3(meta:dict)->tuple[dict,dict]:
    before=unpack(meta);events=copy.deepcopy(before);changes=[]
    # A 174ms first fragment of 'rising' precedes a 349ms rest. Return that
    # fragment to sparks, fixing its second zero-body slot, then start rising after rest.
    index=next((i for i,e in enumerate(events[:-2]) if e["text"]=="rising" and e["kind"]==2
        and events[i+1]["kind"]==1 and events[i+2]["text"]=="rising" and events[i+2]["kind"]==3),None)
    if index is None or index==0 or events[index-1]["text"]!="sparks":
        raise ValueError("chorus_v3 expected the observed sparks/rising/rest/rising pattern")
    original=events[index].copy();previous=events[index-1]
    for field in ["text","phoneme"]:events[index][field]=previous[field]
    events[index]["kind"]=3;events[index+2]["kind"]=2
    changes.append(dict(word="rising",old_onset_seconds=float(original["start"]),new_onset_seconds=float(events[index+2]["start"]),
        reason="Move a fragmented new-word onset after the observed rest; the pre-rest same-pitch fragment becomes sparks continuation.",
        asr_evidence="v2 ASR labels rises at 1.40–2.08s and an extra and at 2.08–2.96s, across the metadata rest.",
        observed_causality=False))
    # These words already have substantial pre-rest duration. Avoid restarting
    # them after the rest; place the next intended word there instead.
    for previous_word,next_word in [("walk","on"),("find","my")]:
        index=next((i for i,e in enumerate(events[:-1]) if i and events[i-1]["kind"]==1
            and e["text"]==previous_word and e["kind"]==3 and events[i+1]["text"]==next_word and events[i+1]["kind"]==2),None)
        if index is None:
            raise ValueError(f"chorus_v3 expected rest/{previous_word}/{next_word} pattern")
        old=events[index+1].copy()
        events[index]["text"]=old["text"];events[index]["phoneme"]=old["phoneme"]
        events[index]["kind"]=2;events[index+1]["kind"]=3
        changes.append(dict(word=next_word,old_onset_seconds=float(old["start"]),new_onset_seconds=float(events[index]["start"]),
            reason=f"Finish {previous_word} before the rest; start {next_word} at the first post-rest note instead of rearticulating {previous_word}.",
            observed_causality=False))
    events=merge_continuations(events)
    invariants=pitch_invariants(before,events)
    result=pack(meta,events)
    result.update(alignment_revision="v3_rest_aware_english_word_boundaries",alignment_status="approximate_alignment",human_reviewed=False)
    audit_before,audit_after=capacity(before),capacity(events)
    if audit_after["zero_body_new_word_onsets"]:
        raise ValueError("Patch introduced an English new-word onset with no phone-body frames")
    return result,dict(recipe="chorus_v3",changes=changes,invariants=invariants,before=audit_before,after=audit_after,
        limits=["Bright has sufficient source phone capacity, so its pronunciation is unchanged.",
            "Storm/I have no zero-body capacity defect; extra oh/where ASR remains unproven and is not fixed by dictionary changes.",
            "My moves to an 8-frame onset with complete phone passes; this trades a shorter onset for removing an interrupted find word.",
            "Removing fragmented word assignments is an evidence-based hypothesis; audible intelligibility must be checked."])


def prepare(args)->dict:
    source=json.loads(args.source.read_text(encoding="utf-8"))
    if len(source)!=1:
        raise ValueError("This recipe accepts the one-segment short chorus experiment only")
    result,patch=chorus_v3(source[0])
    args.out_dir.mkdir(parents=True,exist_ok=True)
    output=args.out_dir/"target_octave_down_v3_metadata.json"
    if output.exists():
        raise FileExistsError(f"Preserve existing experiments; choose a new out-dir: {output}")
    audio.atomic_json(output,[result]);validate_metadata(output)
    patch.update(status="target_patch_prepared_cpu_validation_pending",gpu_used=False,human_reviewed=False,
        source=dict(path=str(args.source.resolve()),sha256=audio.digest(args.source)),
        output=dict(path=str(output.resolve()),sha256=audio.digest(output)),recipe_code_sha256=audio.digest(Path(__file__)),
        fixed_profile=dict(path=str(args.profile.resolve()),sha256=audio.digest(args.profile)),
        asr_evidence=dict(path=str(args.asr_evidence.resolve()),sha256=audio.digest(args.asr_evidence)))
    audio.atomic_json(args.out_dir/"v3_alignment_patch.json",patch)
    return dict(status=patch["status"],output=str(output.resolve()),changes=patch["changes"],
        before_rest_crossings=len(patch["before"]["rest_crossing_word_continuations"]),
        after_rest_crossings=len(patch["after"]["rest_crossing_word_continuations"]),
        before_zero_body_slots=len(patch["before"]["zero_body_english_slots"]),after_zero_body_slots=len(patch["after"]["zero_body_english_slots"]))


def official_audit(args)->dict:
    sys.path.insert(0,str(args.soulx_root.resolve()))
    from soulxsinger.utils.data_processor import DataProcessor
    phoneset=json.loads(args.phoneset.read_text(encoding="utf-8"))
    processor=DataProcessor(480,24000,str(args.phoneset),device="cpu")
    report={}
    for label,path in [("v2",args.source),("v3",args.target)]:
        meta=json.loads(path.read_text(encoding="utf-8"))[0];events=unpack(meta)
        item=processor.process(copy.deepcopy(meta),None);rows=[]
        for event in events:
            start=int(np.round(float(event["start"])*50));end=min(item["mel2note"].shape[-1],int(np.round(float(event["end"])*50)))
            mapping=item["mel2note"][0,start:end];token_ids=item["phoneme"][0,mapping].tolist()
            names=[phoneset[index] for index in token_ids]
            rows.append(dict(text=event["text"],kind=event["kind"],start_seconds=float(event["start"]),end_seconds=float(event["end"]),
                slot_frames=end-start,english_phone_frames=sum(n.startswith("en_") for n in names),
                phone_body_frames=sum(n.startswith("en_") or n=="<SEP>" for n in names),pad_frames=names.count("<PAD>")))
        report[label]=dict(frame_count=item["mel2note"].shape[-1],total_pad_frames=int((item["mel2note"]==0).sum()),
            word_onsets=[row for row in rows if row["kind"]==2],zero_body_english_slots=[row for row,event in zip(rows,events)
                if event["phoneme"].startswith("en_") and not row["phone_body_frames"]],device=str(item["mel2note"].device))
    if report["v2"]["frame_count"]!=report["v3"]["frame_count"] or report["v3"]["zero_body_english_slots"]:
        raise ValueError("CPU validation failed timeline/capacity checks")
    result=dict(status="official_DataProcessor_cpu_validated",gpu_used=False,files=report,
        source_sha256=audio.digest(args.source),target_sha256=audio.digest(args.target),
        warning="Tensor capacity and word allocation improved; this is not an acoustic quality or identity score.")
    audio.atomic_json(args.out,result)
    return result


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    patch=sub.add_parser("chorus-v3")
    for name in ["source","out-dir","profile","asr-evidence"]:patch.add_argument("--"+name,type=Path,required=True)
    audit=sub.add_parser("validate-cpu")
    for name in ["source","target","soulx-root","phoneset","out"]:audit.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();result=prepare(args) if args.command=="chorus-v3" else official_audit(args)
    if args.command=="validate-cpu":
        result={k:v for k,v in result.items() if k!="files"}|{"summary":{k:{key:value for key,value in v.items() if key!="word_onsets"} for k,v in result["files"].items()}}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
