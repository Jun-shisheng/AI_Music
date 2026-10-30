"""Prepare honest approximate English SoulX SVS alignment, using CPU only.

This is a short-sample experiment. It never labels guessed lyric alignment as reviewed.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

from . import audio

CMU_REVISION="74790861f652b15e4ac49015a90074ad62a27690"
CMU_URL=f"https://raw.githubusercontent.com/cmusphinx/cmudict/{CMU_REVISION}/cmudict.dict"
ROOT=Path(__file__).resolve().parents[2]


def read_dictionary(path:Path)->dict[str,list[list[str]]]:
    entries={}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith(";;;"):
            continue
        fields=line.split("#",1)[0].split()
        if len(fields)<2:
            continue
        word=re.sub(r"\(\d+\)$","",fields[0]).lower()
        entries.setdefault(word,[]).append(fields[1:])
    return entries


def phones(word:str,dictionary:dict,phoneset:set)->tuple[str,int,dict]:
    normalized=word.lower().replace("’","'")
    if normalized not in dictionary:
        raise ValueError(f"Word {word!r} is not in the pinned CMU dictionary; supply an explicit pronunciation instead of guessing")
    variants=dictionary[normalized]
    selected=variants[0]
    unknown=["en_"+p for p in selected if "en_"+p not in phoneset]
    if unknown:
        raise ValueError(f"Pronunciation is outside the official SoulX phoneset: {unknown}")
    syllables=sum(bool(re.search(r"[012]$",p)) for p in selected)
    return "en_"+"-".join(selected),max(1,syllables),dict(word=word,cmu_first_pronunciation=selected,
        pronunciation_variants=variants,pronunciation_reviewed=False,syllables=max(1,syllables))


def transcribe(args)->dict:
    # Set before importing CTranslate2/faster-whisper. No CUDA model is loaded.
    os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
    from faster_whisper import WhisperModel
    model=WhisperModel(str(args.asr_model),device="cpu",compute_type="int8",cpu_threads=2,num_workers=1)
    segments,info=model.transcribe(str(args.prompt),language="en",beam_size=5,vad_filter=False,word_timestamps=True,
        condition_on_previous_text=False)
    rows=[]
    for segment in segments:
        rows.append(dict(start=segment.start,end=segment.end,text=segment.text,avg_logprob=segment.avg_logprob,
            no_speech_prob=segment.no_speech_prob,words=[dict(word=w.word,start=w.start,end=w.end,probability=w.probability)
            for w in segment.words or []]))
    result=dict(status="automatic_asr_unreviewed",device="cpu",gpu_used=False,prompt=str(args.prompt.resolve()),
        prompt_sha256=audio.digest(args.prompt),model=str(args.asr_model.resolve()),language=info.language,
        word_timestamps=True,segments=rows,transcript=" ".join(r["text"].strip() for r in rows),
        warning="Sung-word ASR may misrecognize lyrics or timing. No original input lyrics were forced into this transcript.")
    audio.atomic_json(args.out,result)
    return result


def record(items:list[dict],duration:float,name:str,*,f0:np.ndarray|None=None)->dict:
    # Round only to microseconds; coarse 0.01s rounding accumulates across the many ABC notes.
    durations=[float(i["end"]-i["start"]) for i in items]
    if any(d<=0 for d in durations) or abs(sum(durations)-duration)>.0001:
        raise ValueError("Metadata events must cover the entire timeline exactly")
    result=dict(index=name,language="English",time=[0,round(duration*1000)],
        duration=" ".join(f"{d:.6f}" for d in durations),text=" ".join(i["text"] for i in items),
        phoneme=" ".join(i["phoneme"] for i in items),note_pitch=" ".join(str(i["pitch"]) for i in items),
        note_type=" ".join(str(i["type"]) for i in items),alignment_status="approximate_alignment",human_reviewed=False)
    if f0 is not None:
        result["f0"]=" ".join(f"{v:.2f}" for v in f0)
    return result


def prompt_metadata(prompt:Path,contour:Path,asr:dict,dictionary:dict,phoneset:set)->tuple[dict,dict]:
    if audio.digest(prompt)!=asr["prompt_sha256"]:
        raise ValueError("ASR transcript belongs to a different prompt waveform")
    data,rate=audio.read(prompt)
    duration=len(data)/rate
    f0=np.load(contour)
    if abs(len(f0)/50-duration)>.04:
        raise ValueError("Prompt F0 is not aligned on the expected 50Hz grid")
    voiced=f0[f0>0]
    if len(voiced)<10:
        raise ValueError("Prompt has insufficient voiced F0")
    entries=[]
    uncertainties=[]
    def gap(start:float,end:float)->None:
        """Infer a held previous syllable only across initially voiced F0, exposing every change."""
        if end-start<=.0001:
            return
        a,b=int(start*50),min(len(f0),int(end*50))
        mask=f0[a:b]>0
        sustain_end=start
        if entries and entries[-1]["type"]==2 and len(mask) and mask[:min(3,len(mask))].any():
            silence=0
            last_voiced=-1
            for i,voiced_frame in enumerate(mask):
                if voiced_frame:
                    silence=0;last_voiced=i
                else:
                    silence+=1
                    if silence>=5:
                        break
            if last_voiced>=0:
                sustain_end=min(end,(a+last_voiced+1)/50)
            if sustain_end-start>=.08:
                previous=entries[-1]
                uncertainties.append(dict(word=previous["text"],issue="inferred_sustain_beyond_ASR_word_end",
                    asr_end=start,inferred_end=sustain_end,source="DIO initial voiced run allowing gaps shorter than 100ms",
                    human_verified=False))
                previous["end"]=sustain_end
            else:
                sustain_end=start
        if end-sustain_end>.0001:
            entries.append(dict(start=sustain_end,end=end,text="<SP>",phoneme="<SP>",pitch=0,type=1))
    words=[w for s in asr["segments"] for w in s["words"]]
    cursor=0.
    for word in words:
        tokens=re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)*",word["word"])
        if len(tokens)!=1:
            raise ValueError(f"ASR token cannot be mapped safely: {word['word']!r}")
        text=tokens[0]
        start=max(cursor,float(word["start"]))
        end=min(duration,float(word["end"]))
        if end-start<.02:
            uncertainties.append(dict(word=text,issue="zero_or_overlapping_asr_duration",asr=word))
            continue
        if start-cursor>.0001:
            gap(cursor,start)
        phone,_,pronunciation=phones(text,dictionary,phoneset)
        local=f0[int(start*50):max(int(start*50)+1,int(end*50))]
        local=local[local>0]
        pitch_estimate=float(np.median(local)) if len(local) else float(np.median(voiced))
        pitch=int(round(69+12*np.log2(pitch_estimate/440)))
        entries.append(dict(start=start,end=end,text=text,phoneme=phone,pitch=pitch,type=2))
        needed_frames=len(phone[3:].split("-"))+3
        if (end-start)*50<needed_frames:
            uncertainties.append(dict(word=text,issue="word_short_for_phone_frame_count",duration=end-start,
                needed_frames=needed_frames))
        if not len(local):
            uncertainties.append(dict(word=text,issue="no_local_voiced_f0_used_prompt_median"))
        if word["probability"]<.65 or len(pronunciation["pronunciation_variants"])>1:
            uncertainties.append(dict(word=text,issue="asr_or_pronunciation_uncertainty",probability=word["probability"],
                pronunciation_variants=pronunciation["pronunciation_variants"]))
        cursor=end
    if cursor<duration:
        gap(cursor,duration)
    if not any(i["type"]==2 for i in entries):
        raise ValueError("Prompt ASR produced no usable sung words")
    return record(entries,duration,"iron_rain_prompt_approx",f0=f0),dict(events=entries,uncertainties=uncertainties,
        approach="faster_whisper_word_boundaries_plus_explicit_DIO_sustain_gap_inference_CMU_and_local_DIO_median_pitch",
        forced_alignment=False,phoneme_boundaries_observed=False)


KEY_FIFTHS={"C":0,"G":1,"D":2,"A":3,"E":4,"B":5,"F#":6,"C#":7,"F":-1,"Bb":-2,"Eb":-3,"Ab":-4,"Db":-5,"Gb":-6,"Cb":-7,
    "Am":0,"Em":1,"Bm":2,"F#m":3,"C#m":4,"G#m":5,"D#m":6,"A#m":7,"Dm":-1,"Gm":-2,"Cm":-3,"Fm":-4,"Bbm":-5,"Ebm":-6,"Abm":-7}


def abc_notes(path:Path)->tuple[list[list[dict]],dict]:
    from tools.reference_studio import plan
    header,sections=plan.parse(path.read_text(encoding="utf-8-sig"))
    clock=plan.timing(header)
    fields={line.split(":",1)[0]:line.split(":",1)[1].strip() for line in header if ":" in line}
    key=fields["K"]
    if key not in KEY_FIFTHS:
        raise ValueError(f"Unsupported ABC key {key!r}")
    fifths=KEY_FIFTHS[key]
    accidentals={letter:1 for letter in "FCGDAEB"[:fifths]} if fifths>=0 else {letter:-1 for letter in "BEADGCF"[:-fifths]}
    default=Fraction(fields["L"])
    quarter_seconds=60/clock["quarter_bpm"]
    result=[]
    cursor=0.
    previous=None
    for section in sections:
        for chunk in section.chunks:
            for field in plan.bars(chunk.vocal,clock["units_per_bar"]):
                bar=[]
                carried={}
                for token in plan.TOKEN.finditer(field):
                    text=token[0]
                    if text.startswith('"'):
                        continue
                    if text.startswith(("z","x")):
                        units=plan.length(text[1:]);pitch=0;tie=False
                    else:
                        match=plan.NOTE.fullmatch(text)
                        name,suffix,tie=match.groups()
                        detail=re.fullmatch(r"(\^+|_+|=)?([A-Ga-g])([,']*)",name)
                        accidental,letter,octaves=detail.groups()
                        octave=(5 if letter.islower() else 4)+octaves.count("'")-octaves.count(",")
                        pitch=12*(octave+1)+dict(C=0,D=2,E=4,F=5,G=7,A=9,B=11)[letter.upper()]
                        identity=(letter.upper(),octave)
                        if accidental:
                            carried[identity]=accidental.count("^")-accidental.count("_")
                        pitch+=carried.get(identity,accidentals.get(letter.upper(),0))
                        units=plan.length(suffix)
                    duration=float(units*default/Fraction(1,4))*quarter_seconds
                    note=dict(start=cursor,end=cursor+duration,pitch=pitch,abc_token=text,tie_to_next=bool(tie),bar=len(result))
                    if previous and previous["tie_to_next"] and previous["pitch"]!=pitch:
                        raise ValueError("ABC tie leads to a different pitch")
                    bar.append(note);previous=note;cursor+=duration
                result.append(bar)
    return result,dict(key=key,duration_seconds=cursor,bars=len(result),tempo_bpm=clock["quarter_bpm"],
        interpretation="ABC standard absolute pitches; YuE audio may use another octave")


def target_metadata(bars:list[list[dict]],lines:list[str],dictionary:dict,phoneset:set,shift:int)->tuple[dict,dict]:
    if len(bars)%len(lines):
        raise ValueError("Cannot distribute lyric lines into an integral equal number of bars; provide manual alignment")
    per_line=len(bars)//len(lines)
    entries=[]
    assignments=[]
    tie_carry_notes=0
    for index,line in enumerate(lines):
        words=re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)*",line)
        if not words:
            raise ValueError("Empty lyric line")
        pronounced=[phones(w,dictionary,phoneset) for w in words]
        notes=[n for bar in bars[index*per_line:(index+1)*per_line] for n in bar]
        voiced=[n for n in notes if n["pitch"]]
        # Ties must keep one lyric identity; collapse tie chains into allocation units.
        groups=[]
        for note in voiced:
            if groups and groups[-1][-1]["tie_to_next"]:
                groups[-1].append(note)
            else:
                groups.append([note])
        carried=[]
        first_bar=index*per_line
        if first_bar and bars[first_bar-1][-1]["tie_to_next"]:
            if not entries or entries[-1]["type"]==1:
                raise ValueError("Cannot carry a lyric over a cross-line tie after a rest")
            carried=groups.pop(0)
            tie_carry_notes+=len(carried)
        if len(groups)<len(words):
            raise ValueError(f"Line {index+1} has {len(words)} words but only {len(groups)} untied note groups")
        total_voiced=sum(n["end"]-n["start"] for group in groups for n in group)
        weights=np.array([p[1] for p in pronounced],dtype=float)
        desired=np.cumsum(weights)/weights.sum()*total_voiced
        elapsed=0.;word_index=0;seen=set();mapping={}
        for note in carried:
            previous=entries[-1]
            mapping[id(note)]=dict(start=note["start"],end=note["end"],text=previous["text"],phoneme=previous["phoneme"],
                pitch=note["pitch"]+shift,type=3)
        for group_index,group in enumerate(groups):
            # At least one allocation unit per remaining word; otherwise nearest syllable-weight boundary.
            remaining_groups=len(groups)-group_index
            remaining_words=len(words)-word_index
            if word_index<len(words)-1 and word_index in seen and (elapsed>=desired[word_index] or remaining_groups==remaining_words-1):
                word_index+=1
            for note in group:
                first=word_index not in seen
                seen.add(word_index)
                event=dict(start=note["start"],end=note["end"],text=words[word_index],phoneme=pronounced[word_index][0],
                    pitch=note["pitch"]+shift,type=2 if first else 3)
                mapping[id(note)]=event
                elapsed+=note["end"]-note["start"]
        if len(seen)!=len(words):
            raise ValueError(f"Could not assign every lyric word in line {index+1}; use manual alignment")
        for note in notes:
            entries.append(mapping[id(note)] if note["pitch"] else dict(start=note["start"],end=note["end"],
                text="<SP>",phoneme="<SP>",pitch=0,type=1))
        assignments.append(dict(line=index+1,lyrics=line,start_seconds=notes[0]["start"],end_seconds=notes[-1]["end"],
            words=[dict(**p[2],assigned_note_count=sum(e["text"]==word for e in mapping.values())) for word,p in zip(words,pronounced)]))
    duration=entries[-1]["end"]
    merged=[]
    for entry in entries:
        if merged and entry["type"]==3 and all(entry[k]==merged[-1][k] for k in ["text","phoneme","pitch"]):
            merged[-1]["end"]=entry["end"]
        else:
            merged.append(entry.copy())
    short=[dict(text=e["text"],start=e["start"],end=e["end"],phone_count=len(e["phoneme"][3:].split("-")))
        for e in merged if e["type"]!=1 and (e["end"]-e["start"])*50<len(e["phoneme"][3:].split("-"))+3]
    return record(merged,duration,"new_lyrics_chorus_approx"),dict(line_assignments=assignments,
        same_pitch_same_word_continuations_merged=len(entries)-len(merged),cross_line_tied_notes_kept_with_previous_word=tie_carry_notes,
        short_phone_frame_warnings=short,
        target_register_adaptation_semitones=shift,relative_melody_preserved=True,absolute_pitch_preserved=shift==0,
        method="equal_bar_groups_per_line_then_CMU_syllable_weighted_note_group_allocation",
        word_to_note_alignment_observed=False,warning="This is a proposed singable allocation, not recovered or manually corrected lyric-to-note alignment.")


def prepare(args)->dict:
    args.out_dir.mkdir(parents=True,exist_ok=True)
    dictionary=read_dictionary(args.dictionary)
    phoneset=set(json.loads(args.phoneset.read_text(encoding="utf-8")))
    transcript=json.loads(args.asr_json.read_text(encoding="utf-8"))
    prompt,prompt_report=prompt_metadata(args.prompt,args.prompt_f0,transcript,dictionary,phoneset)
    bars,clock=abc_notes(args.score)
    lines=[line.strip() for line in args.lyrics.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not re.fullmatch(r"\[.*\]",line.strip())]
    native,native_report=target_metadata(bars,lines,dictionary,phoneset,0)
    adapted,adapted_report=target_metadata(bars,lines,dictionary,phoneset,-12)
    for name,value in [("prompt_metadata.json",prompt),("target_native_metadata.json",native),("target_octave_down_metadata.json",adapted)]:
        audio.atomic_json(args.out_dir/name,[value])
    from .workflow import validate_metadata
    for name in ["prompt_metadata.json","target_native_metadata.json","target_octave_down_metadata.json"]:
        validate_metadata(args.out_dir/name,prompt=name.startswith("prompt"))
    result=dict(status="approximate_alignment_prepared",human_reviewed=False,listening_approved=False,gpu_used=False,
        source_files={name:dict(path=str(path.resolve()),sha256=audio.digest(path)) for name,path in
            dict(prompt=args.prompt,prompt_f0=args.prompt_f0,asr=args.asr_json,score=args.score,lyrics=args.lyrics,
                 dictionary=args.dictionary,phoneset=args.phoneset).items()},
        dictionary_source=dict(url=CMU_URL,revision=CMU_REVISION),clock=clock,prompt_alignment=prompt_report,
        target_native=native_report,target_octave_down=adapted_report,
        official_conventions=dict(english="en_ plus hyphen-joined ARPAbet with stress",note_type={"1":"silence","2":"new lyric note","3":"lyric continuation/melisma"}),
        uncertainties=["Prompt ASR may mishear sung English; word timing is automatic, not forced phoneme alignment.",
            "Prompt pitch per word uses median voiced DIO frames; pitch changes inside a word are approximated.",
            "Target has 4 lines assigned to 8 bars by design; no ground-truth word/note timing exists.",
            "Repeated note_type=3 continuation is an official representation but intelligibility requires listening.",
            "CMU first pronunciation may differ from desired singing pronunciation.",
            "Octave-down version changes absolute register by -12 semitones; pitch classes and relative melody stay the same."])
    audio.atomic_json(args.out_dir/"preparation_manifest.json",result)
    return result


def validate_cpu(args)->dict:
    sys.path.insert(0,str(args.soulx_root.resolve()))
    import torch
    from soulxsinger.utils.data_processor import DataProcessor
    processor=DataProcessor(480,24000,str(args.phoneset),device="cpu")
    results={}
    for path in [args.out_dir/"prompt_metadata.json",args.out_dir/"target_native_metadata.json",args.out_dir/"target_octave_down_metadata.json"]:
        metadata=json.loads(path.read_text(encoding="utf-8"))[0]
        item=processor.process(metadata,str(args.prompt) if path.name.startswith("prompt") else None)
        mapping=item["mel2note"]
        results[path.name]=dict(frame_count=mapping.shape[-1],duration_seconds=mapping.shape[-1]/50,
            token_count=item["phoneme"].shape[-1],unassigned_frames=int((mapping==0).sum()),
            max_mapping_index=int(mapping.max()),last_frame_mapping=int(mapping[0,-1]),
            waveform_samples=item["waveform"].shape[-1] if "waveform" in item else None,
            note_pitch_min=int(item["note_pitch"][item["note_pitch"]>0].min()),
            note_pitch_max=int(item["note_pitch"].max()),device=str(mapping.device))
        if not torch.isfinite(mapping).all() or int(mapping.max())>=item["phoneme"].shape[-1]:
            raise ValueError("Official DataProcessor produced invalid note/token indices")
    result=dict(status="official_DataProcessor_cpu_validated",gpu_used=False,human_alignment_reviewed=False,files=results,
        warning="Tensor compatibility confirms that the model can consume metadata; it does not validate lyric timing or pronunciation.")
    audio.atomic_json(args.out_dir/"cpu_validation.json",result)
    return result


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    asr=commands.add_parser("asr")
    asr.add_argument("--prompt",type=Path,required=True);asr.add_argument("--asr-model",type=Path,required=True);asr.add_argument("--out",type=Path,required=True)
    prep=commands.add_parser("prepare")
    for name in ["prompt","prompt-f0","asr-json","score","lyrics","dictionary","phoneset","out-dir"]:
        prep.add_argument("--"+name,type=Path,required=True)
    check=commands.add_parser("validate-cpu")
    for name in ["out-dir","prompt","phoneset","soulx-root"]:
        check.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    result=transcribe(args) if args.command=="asr" else prepare(args) if args.command=="prepare" else validate_cpu(args)
    print(json.dumps({k:v for k,v in result.items() if k in {"status","gpu_used","clock","transcript","files","warning"}},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
