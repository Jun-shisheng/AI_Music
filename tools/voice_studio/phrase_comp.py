"""CPU-only phrase comp at a measured shared low-energy window, without changing timing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import audio
from .svs_alignment import unpack,pitch_invariants


class NoSafeCut(ValueError):
    pass


def validate_pair(early:np.ndarray,late:np.ndarray)->None:
    if early.shape!=late.shape or early.ndim!=2:
        raise ValueError("Comp parents must have exactly the same frame count and channel layout")
    if not len(early) or not np.isfinite(early).all() or not np.isfinite(late).all():
        raise ValueError("Comp parents must contain finite, nonempty audio")


def choose_cut(early:np.ndarray,late:np.ndarray,rate:int,*,center_seconds:float,radius_seconds:float,
               fade_ms:float=20,max_rms_db:float=-34,max_window_peak:float=.08)->dict:
    validate_pair(early,late)
    if rate<=0 or not 5<=fade_ms<=100 or radius_seconds<0:
        raise ValueError("Invalid sample rate, short fade duration, or search radius")
    width=round(fade_ms/1000*rate)
    if width<2:
        raise ValueError("Fade must contain at least two samples")
    center=round(center_seconds*rate);radius=round(radius_seconds*rate)
    lo=max(width//2,center-radius);hi=min(len(early)-(width-width//2),center+radius)
    if lo>hi or center<0 or center>len(early):
        raise NoSafeCut("Search center has no complete fade window inside the parent audio")
    allowed_rms=10**(max_rms_db/20)
    candidates=[]
    for cut in range(lo,hi+1,max(1,round(rate*.001))):
        start=cut-width//2;end=start+width
        a,b=early[start:end],late[start:end]
        levels=[float(np.sqrt(np.mean(x.astype(np.float64)**2))) for x in (a,b)]
        peaks=[float(np.max(np.abs(x))) for x in (a,b)]
        if max(levels)<=allowed_rms and max(peaks)<=max_window_peak:
            candidates.append((max(levels),abs(cut-center),cut,start,end,levels,peaks))
    if not candidates:
        raise NoSafeCut("No shared low-energy fade window satisfies both RMS and peak limits; refusing to hard splice")
    _,_,cut,start,end,levels,peaks=min(candidates)
    return dict(cut_sample=cut,cut_seconds=cut/rate,fade_start_sample=start,fade_end_sample=end,
        fade_start_seconds=start/rate,fade_end_seconds=end/rate,fade_samples=width,fade_ms=width/rate*1000,
        early_window_rms=levels[0],late_window_rms=levels[1],
        early_window_rms_dbfs=20*np.log10(max(levels[0],1e-12)),late_window_rms_dbfs=20*np.log10(max(levels[1],1e-12)),
        early_window_peak=peaks[0],late_window_peak=peaks[1],rms_limit_dbfs=max_rms_db,window_peak_limit=max_window_peak,
        search_center_seconds=center_seconds,search_radius_seconds=radius_seconds,
        selection="minimum worst-parent RMS among windows passing both limits; ties prefer the requested boundary")


def crossfade(early:np.ndarray,late:np.ndarray,start:int,end:int)->np.ndarray:
    validate_pair(early,late)
    if not 0<start<end<len(early) or end-start<2:
        raise ValueError("Crossfade must be a complete interior interval with at least two samples")
    output=early.copy()
    alpha=np.linspace(0.,1.,end-start,dtype=np.float64)[:,None]
    output[start:end]=(early[start:end]*(1-alpha)+late[start:end]*alpha).astype(np.float32)
    output[end:]=late[end:]
    if len(output)!=len(early) or not np.isfinite(output).all():
        raise ValueError("Comp lost its timeline or produced nonfinite samples")
    return output


def comp(args)->dict:
    if args.out_dir.exists() and (not args.out_dir.is_dir() or any(args.out_dir.iterdir())):
        raise FileExistsError("Output destination already contains data; use a new output directory to preserve existing candidates")
    early,early_rate=audio.read(args.early);late,late_rate=audio.read(args.late)
    validate_pair(early,late)
    if early_rate!=late_rate or early_rate!=args.sample_rate:
        raise ValueError("Parent sample rates differ from each other or the requested sample rate")
    if len(early)!=round(args.expected_seconds*args.sample_rate):
        raise ValueError("Parent duration differs from the requested exact output timeline")
    a=json.loads(args.early_score.read_text(encoding="utf-8"));b=json.loads(args.late_score.read_text(encoding="utf-8"))
    if len(a)!=1 or len(b)!=1 or a[0]["time"]!=b[0]["time"]:
        raise ValueError("Parent scores do not share one output timeline")
    invariants=pitch_invariants(unpack(a[0]),unpack(b[0]))
    cut=choose_cut(early,late,early_rate,center_seconds=args.center,radius_seconds=args.radius,
        fade_ms=args.fade_ms,max_rms_db=args.max_rms_db,max_window_peak=args.max_window_peak)
    output=crossfade(early,late,cut["fade_start_sample"],cut["fade_end_sample"])
    args.out_dir.mkdir(parents=True,exist_ok=True)
    vocals=args.out_dir/"comp_vocals.wav";lyrics=args.out_dir/"intended_lyrics.txt"
    audio.write(vocals,output,early_rate);shutil.copy2(args.lyrics,lyrics)
    native,rate=audio.read(vocals)
    if native.shape!=early.shape or rate!=early_rate:
        raise ValueError("Exported comp changed shape or sample rate")
    # PCM16 parents can be preserved bit-for-bit outside the fade because there is no gain processing.
    if not np.array_equal(native[:cut["fade_start_sample"]],early[:cut["fade_start_sample"]]) or not np.array_equal(native[cut["fade_end_sample"]:],late[cut["fade_end_sample"]:]):
        raise ValueError("Export altered a parent region outside the crossfade")
    result=dict(schema_version=1,kind="phrase_comp",status="awaiting_listening",inference_performed=False,gpu_used=False,
        listening_approved=False,subjective_quality_score=None,singer_identity_score=None,
        helper_source=dict(path=str(Path(__file__).resolve()),sha256=audio.digest(Path(__file__))),
        request=dict(early=str(args.early.resolve()),late=str(args.late.resolve()),
            early_score=str(args.early_score.resolve()),late_score=str(args.late_score.resolve()),
            lyrics=str(args.lyrics.resolve()),out_dir=str(args.out_dir.resolve()),
            center_seconds=args.center,radius_seconds=args.radius,fade_ms=args.fade_ms,
            max_rms_dbfs=args.max_rms_db,max_window_peak=args.max_window_peak,
            sample_rate=args.sample_rate,expected_seconds=args.expected_seconds,boundary_note=args.boundary_note),
        parents={label:dict(path=str(path.resolve()),sha256=audio.digest(path)) for label,path in [("early",args.early),("late",args.late)]},
        score_provenance={label:dict(path=str(path.resolve()),sha256=audio.digest(path)) for label,path in [("early",args.early_score),("late",args.late_score)]},
        intended_lyrics=dict(path=str(lyrics.resolve()),sha256=audio.digest(lyrics)),
        outputs=dict(vocals=dict(path=str(vocals.resolve()),sha256=audio.digest(vocals))),
        audio=dict(frames=len(native),sample_rate=rate,channels=native.shape[1],duration_seconds=len(native)/rate,
            sample_peak=float(np.max(np.abs(native))),nonfinite_samples=int((~np.isfinite(native)).sum()),
            near_full_scale_samples=int((np.abs(native)>=.999).sum())),
        cut=cut,fade_curve="linear complementary weights, exactly 1.0 combined weight",
        invariants={**invariants,"sample_count_preserved":True,"no_time_stretch":True,"no_global_or_dynamic_gain":True,
            "early_region_bit_exact_outside_fade":True,"late_region_bit_exact_outside_fade":True},
        boundary_interpretation=args.boundary_note,
        limits=["A low-energy shared window reduces splice risk but is not a listening verdict.",
            "Metadata phrase times and ASR word times are approximate and do not prove a true inter-word pause.",
            "No new singing was inferred and no singer-identity or subjective-quality score was assigned."])
    audio.atomic_json(args.out_dir/"manifest.json",result)
    return result


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["early","late","early-score","late-score","lyrics","out-dir"]:parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--center",type=float,default=11.162796);parser.add_argument("--radius",type=float,default=.65)
    parser.add_argument("--fade-ms",type=float,default=20);parser.add_argument("--max-rms-db",type=float,default=-34)
    parser.add_argument("--max-window-peak",type=float,default=.08);parser.add_argument("--sample-rate",type=int,default=24000)
    parser.add_argument("--expected-seconds",type=float,default=22.326)
    parser.add_argument("--boundary-note",default="Shared low-energy window near the user-specified location; it does not establish a phrase or word boundary.")
    result=comp(parser.parse_args())
    print(json.dumps({k:v for k,v in result.items() if k in ["kind","status","inference_performed","outputs","audio","cut","boundary_interpretation"]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
