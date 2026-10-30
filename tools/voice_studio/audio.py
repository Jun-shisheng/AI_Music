"""Small CPU-only audio utilities. Inference lives in a separate interpreter."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

RATE = 24000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError(f"Expected canonical PCM16 WAV: {path}")
        data = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768
        return data.reshape(-1, f.getnchannels()), f.getframerate()


def write(path: Path, data: np.ndarray, rate: int = RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    if not np.isfinite(data).all():
        raise ValueError("Audio contains NaN or infinite samples")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(data.shape[1])
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(np.rint(np.clip(data, -1, 32767 / 32768) * 32768).astype("<i2").tobytes())


def ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        raise FileNotFoundError("ffmpeg is not on PATH")
    return found


def info(path: Path) -> dict:
    ffprobe=shutil.which("ffprobe")
    if not ffprobe:
        raise FileNotFoundError("ffprobe is not on PATH")
    completed=subprocess.run([ffprobe,"-v","error","-select_streams","a:0","-show_entries",
        "stream=sample_rate,channels,duration:format=duration","-of","json",str(path)],check=True,capture_output=True,text=True)
    metadata=json.loads(completed.stdout)
    stream=metadata["streams"][0]
    return dict(sample_rate=int(stream["sample_rate"]),channels=int(stream["channels"]),
        duration_seconds=float(stream.get("duration",metadata["format"]["duration"])))


def canonical_at(source: Path, target: Path, *, rate: int, start: float = 0, seconds: float = 0,
                 channels: int = 1) -> Path:
    """Convert to canonical PCM16 at an explicit rate (24k for SVC, 44.1k for delivery)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg(), "-nostdin", "-v", "error", "-y", "-i", str(source)]
    if start:
        cmd += ["-ss", str(start)]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-ar", str(rate), "-ac", str(channels), "-c:a", "pcm_s16le", str(target)]
    subprocess.run(cmd, check=True, capture_output=True)
    return target


def canonical(source: Path, target: Path, *, start: float = 0, seconds: float = 0, channels: int = 1) -> Path:
    return canonical_at(source, target, rate=RATE, start=start, seconds=seconds, channels=channels)


def rms(data: np.ndarray, hop: int = 1200) -> np.ndarray:
    mono = data.mean(axis=1) if data.ndim == 2 else data
    count = math.ceil(len(mono) / hop)
    padded = np.pad(mono, (0, count * hop - len(mono)))
    return np.sqrt(np.mean(padded.reshape(count, hop) ** 2, axis=1) + 1e-12)


def choose_reference(data: np.ndarray, rate: int, seconds: float = 8) -> tuple[int, int]:
    """Choose an active, stable excerpt; final singer identity is a listening decision."""
    if len(data) < 3 * rate:
        raise ValueError("Reference vocal must contain at least 3 seconds")
    size = min(len(data), round(seconds * rate))
    best = (float("-inf"), 0)
    for start in range(0, len(data) - size + 1, max(1, rate // 2)):
        levels = rms(data[start:start + size])
        active = np.mean(levels > 0.012)
        score = active - float(np.std(np.log(levels + 1e-4))) * 0.07
        if score > best[0]:
            best = score, start
    return best[1], best[1] + size


def reference_candidates(data: np.ndarray, rate: int, seconds: float = 8, count: int = 3) -> list[dict]:
    if not 3 <= seconds <= 15:
        raise ValueError("Reference length must be between 3 and 15 seconds")
    size = min(len(data), round(seconds * rate))
    ranked = []
    for start in range(0, len(data)-size+1, max(1,rate//2)):
        levels = rms(data[start:start+size])
        active = float(np.mean(levels > .012))
        stability = float(np.std(np.log(levels+1e-4)))
        score = active-stability*.07
        ranked.append(dict(start=start,end=start+size,active_ratio=round(active,4),
            rms_dbfs=round(float(20*np.log10(np.sqrt(np.mean(data[start:start+size]**2))+1e-8)),2),
            selection_score=round(score,4)))
    chosen=[]
    for candidate in sorted(ranked,key=lambda x:x["selection_score"],reverse=True):
        if all(abs(candidate["start"]-old["start"])>=size for old in chosen):
            chosen.append(candidate)
            if len(chosen)==count:
                break
    if not chosen:
        raise ValueError("Reference vocal must contain at least 3 seconds")
    return chosen


def segments(data: np.ndarray, rate: int = RATE, max_seconds: float = 12, context_seconds: float = .18) -> list[dict]:
    if not 3 <= max_seconds <= 25:
        raise ValueError("chunk_seconds must be between 3 and 25")
    if not 0 <= context_seconds < max_seconds / 4:
        raise ValueError("Invalid context duration")
    levels = rms(data, max(1, rate // 20))
    length = len(data)
    cuts = [0]
    while length - cuts[-1] > max_seconds * rate:
        # Prefer a breath or silence near the upper chunk bound, never make oversized chunks.
        hi = min(len(levels) - 1, int((cuts[-1] / rate + max_seconds) * 20))
        lo = max(hi - 40, int(cuts[-1] / rate * 20) + 20)
        cut = (lo + int(np.argmin(levels[lo:hi + 1]))) * (rate // 20)
        cuts.append(max(cuts[-1] + rate, min(cut, length)))
    cuts.append(length)
    context = round(context_seconds * rate)
    return [dict(index=i, start=a, end=b, context_start=max(0, a-context),
                 context_end=min(length, b+context)) for i, (a, b) in enumerate(zip(cuts, cuts[1:]))]


def voiced_segments(data: np.ndarray, rate: int, f0: np.ndarray, *, min_seconds: float = 6.0,
                    max_seconds: float = 18.0, context_seconds: float = .25,
                    search_seconds: float = 3.0, max_stretch: float = 1.25,
                    forced_context_seconds: float = .5) -> list[dict]:
    """Cut at long unvoiced runs instead of a fixed clock.

    SoulX-SVC's own long-audio path splits at >=10 unvoiced frames and keeps
    segments 15-30 s. The old workflow cut every 12 s at the lowest RMS in the
    last 2 s, which lands inside sustained high notes and makes the model
    restart the phrase. This segmenter returns the same entry schema as
    `segments()` so `merge()` and its exact-timeline invariant stay untouched.

    `f0` is the 50 Hz model contour; voiced frames are `f0 > 0`.
    """
    if not 3 <= max_seconds <= 30:
        raise ValueError("max_seconds must be between 3 and 30")
    if not 0 < min_seconds < max_seconds:
        raise ValueError("min_seconds must be positive and below max_seconds")
    if not 0 <= context_seconds < min_seconds / 2:
        raise ValueError("Invalid context duration")
    frame_rate = 50.0
    levels = rms(data, max(1, rate // 50))
    voiced = np.zeros(len(levels), dtype=bool)
    contour = np.asarray(f0).reshape(-1)
    voiced[:min(len(voiced), len(contour))] = contour[:len(voiced)] > 0
    length = len(data)

    def frame_to_sample(frame: int) -> int:
        return int(round(frame / frame_rate * rate))

    cuts = [0]
    reasons: list[dict] = []
    forced_contexts: dict[int, float] = {}
    while length - cuts[-1] > max_seconds * rate:
        start_frame = int(round(cuts[-1] / rate * frame_rate))
        min_frame = start_frame + int(round(min_seconds * frame_rate))
        max_frame = start_frame + int(round(max_seconds * frame_rate))
        hi = min(len(levels) - 1, max_frame)
        lo = max(min_frame, hi - int(round(search_seconds * frame_rate)))
        if hi <= lo:
            lo = min_frame
        window = np.arange(lo, hi + 1)

        def pick(candidates: np.ndarray) -> tuple[int, dict] | None:
            best = None
            for index in candidates:
                left, right = max(0, index - 2), min(len(voiced), index + 3)
                ratio = float(np.mean(voiced[left:right])) if right > left else 0.0
                if ratio > 0.4:
                    continue
                level = float(levels[index])
                score = -20.0 * np.log10(level + 1e-9) - 6.0 * ratio
                if best is None or score > best[0]:
                    best = (score, dict(frame=int(index), voiced_ratio=ratio, rms=level,
                                        reason="rest" if ratio == 0 else "low_energy"))
            return (best[1]["frame"], best[1]) if best else None

        chosen = pick(window)
        if chosen is None:
            extended_hi = min(len(levels) - 1, start_frame + int(round(max_seconds * max_stretch * frame_rate)))
            chosen = pick(np.arange(lo, extended_hi + 1))
        if chosen is None:
            index = int(window[int(np.argmin(voiced[window]))])
            chosen = (index, dict(frame=index, voiced_ratio=float(np.mean(voiced[max(0, index - 2):index + 3])),
                                  rms=float(levels[index]), reason="forced_voiced_cut"))
        index, info = chosen
        cut = max(cuts[-1] + int(min_seconds * rate), min(frame_to_sample(index), length))
        cuts.append(cut)
        reasons.append(info)
        if info["reason"] == "forced_voiced_cut":
            # The cut closes the chunk whose index is len(cuts) - 2.
            forced_contexts[len(cuts) - 2] = forced_context_seconds

    if cuts[-1] != length:
        cuts.append(length)
    if len(cuts) >= 3 and length - cuts[-2] < min_seconds * rate:
        moved = length - int(min_seconds * rate)
        if moved - cuts[-3] >= min_seconds * rate:
            cuts[-2] = moved
            reasons[-1] = dict(frame=int(round(moved / rate * frame_rate)), voiced_ratio=None,
                               rms=None, reason="tail_rebalanced")
        else:
            cuts.pop(-2)
            reasons.pop()
            forced_contexts.clear()
            reasons[-1] = dict(frame=None, voiced_ratio=None, rms=None, reason="tail_merged")

    entries = []
    for index, (start, end) in enumerate(zip(cuts, cuts[1:])):
        context = round(forced_contexts.get(index, context_seconds) * rate)
        entries.append(dict(index=index, start=start, end=end,
                            context_start=max(0, start - context),
                            context_end=min(length, end + context),
                            cut_reason=reasons[index]["reason"] if index < len(reasons) else "tail",
                            cut_voiced_ratio=reasons[index]["voiced_ratio"] if index < len(reasons) else None,
                            cut_rms=reasons[index]["rms"] if index < len(reasons) else None,
                            chunk_seconds=round((end - start) / rate, 3),
                            context_seconds=context / rate,
                            forced=bool(forced_contexts.get(index))))
    return entries


def merge(parts: list[tuple[dict, np.ndarray]], length: int) -> np.ndarray:
    result = np.zeros(length, dtype=np.float64)
    weights = np.zeros(length, dtype=np.float64)
    for segment, audio in parts:
        a, b = segment["context_start"], segment["context_end"]
        samples = np.asarray(audio).reshape(-1)
        if len(samples) != b - a:
            raise ValueError(f"Chunk {segment['index']} duration mismatch: {len(samples)} != {b-a}")
        w = np.ones(len(samples))
        left, right = segment["start"] - a, b - segment["end"]
        if left:
            w[:left] = np.linspace(0, 1, left, endpoint=False)
        if right:
            w[-right:] = np.linspace(1, 0, right, endpoint=False)
        result[a:b] += samples * w
        weights[a:b] += w
    if (weights == 0).any():
        raise ValueError("Chunk reconstruction left a gap in the timeline")
    return (result / weights).astype(np.float32)


def match_vocal_level(source: np.ndarray, generated: np.ndarray, rate: int = RATE) -> tuple[np.ndarray,dict]:
    """One gain for the entire clip, measured only where the source actually sings."""
    source=np.asarray(source).reshape(-1)
    generated=np.asarray(generated).reshape(-1)
    if len(source)!=len(generated) or not np.isfinite(generated).all():
        raise ValueError("Level matching requires finite vocals on the same timeline")
    hop=max(1,rate//20)
    levels=rms(source,hop)
    threshold=max(.008,float(np.percentile(levels,90))*.15)
    active=np.repeat(levels>threshold,hop)[:len(source)]
    source_rms=float(np.sqrt(np.mean(source[active]**2))) if active.any() else 0.
    generated_rms=float(np.sqrt(np.mean(generated[active]**2))) if active.any() else 0.
    gain_db=0.
    reason="matched_source_active_rms"
    requested_db=0.
    if active.sum()<rate*.25 or source_rms<.01:
        reason="source_too_quiet_or_short_to_match"
    elif generated_rms<.001:
        reason="generated_vocal_too_quiet_to_boost"
    else:
        requested_db=float(20*np.log10(source_rms/generated_rms))
        gain_db=float(np.clip(requested_db,-6,6))
    peak=float(np.max(np.abs(generated))) if len(generated) else 0.
    if peak:
        # A constant peak bound protects the dry export without time-varying processing.
        gain_db=min(gain_db,float(20*np.log10(.98/peak)))
    gain=float(10**(gain_db/20))
    matched=(generated*gain).astype(np.float32)
    return matched,dict(method="constant_gain_source_active_rms",gain_db=round(gain_db,6),linear_gain=gain,
        requested_gain_db=round(requested_db,6),reason=reason,source_active_rms=source_rms,
        generated_active_rms=generated_rms,source_activity_threshold=threshold,
        active_seconds=round(float(active.sum())/rate,3),rms_match_range_db=[-6,6],
        gain_limited=abs(requested_db-gain_db)>1e-5,dry_peak_limit=.98,time_varying_gain=False)


def static_mix(instrumental:np.ndarray,vocals:np.ndarray,length:int,vocal_gain_db:float=0.)->tuple[np.ndarray,dict]:
    background=np.asarray(instrumental,dtype=np.float64)
    if background.ndim==1:
        background=background[:,None]
    voice=np.asarray(vocals,dtype=np.float64).reshape(-1)
    if len(voice)!=length:
        raise ValueError("Dry vocal and requested mix durations differ")
    original_length=len(background)
    if original_length<length:
        background=np.pad(background,((0,length-original_length),(0,0)))
    background=background[:length]
    vocal_gain=float(10**(vocal_gain_db/20))
    summed=background+voice[:,None]*vocal_gain
    if not np.isfinite(summed).all():
        raise ValueError("Mix contains nonfinite samples")
    peak=float(np.max(np.abs(summed))) if len(summed) else 0.
    global_gain=min(1.,.95/peak) if peak else 1.
    return (summed*global_gain).astype(np.float32),dict(method="sum_then_constant_peak_headroom",
        vocal_relative_gain_db=vocal_gain_db,vocal_relative_linear_gain=vocal_gain,
        global_linear_gain=global_gain,global_gain_db=round(float(20*np.log10(global_gain)),6),
        peak_before=peak,peak_after=peak*global_gain,peak_limit=.95,time_varying_gain=False,
        instrumental_padded_samples=max(0,length-original_length),instrumental_trimmed_samples=max(0,original_length-length))


def technical_qa(source: np.ndarray, output: np.ndarray, rate: int = RATE) -> dict:
    if len(source) != len(output):
        raise ValueError("Output lost timeline alignment")
    hop = max(1, rate // 20)
    a, b = rms(source, hop), rms(output, hop)
    active = a > .015
    dropout = float(np.mean(b[active] < np.maximum(.002, a[active] * .04))) if active.any() else 0
    clipping = float(np.mean(np.abs(output) >= .999))
    warnings = []
    if dropout > .08:
        warnings.append("Possible missing syllables: inspect source-active/output-quiet intervals")
    if clipping > .001:
        warnings.append("Generated vocal contains clipping")
    if not active.any():
        warnings.append("Source has little active vocal content")
    return dict(duration_seconds=round(len(output)/rate, 3), dropout_ratio=round(dropout, 5),
                clipping_ratio=round(clipping, 5), warnings=warnings,
                listening_required=True, identity_score=None,
                note="These checks detect technical failures; they do not grade timbre similarity or singing quality.")
