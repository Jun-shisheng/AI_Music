"""CPU mix descriptors and transparent, linear loudness matching.

Frequency-band and transient activity are proxies, never instrument labels.
There is deliberately no chroma-average hook score or perceptual quality score.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import subprocess

import numpy as np


def decode(path: Path, sr=22050, start=0.0, duration=None) -> np.ndarray:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-i", str(path)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    channels = probe(path)["channels"]
    if channels < 1:
        raise ValueError("Audio has no channels")
    # ffmpeg's automatic stereo downmix uses equal-power coefficients (~.707
    # each), which boosts correlated stereo by 3dB. Match our loudness code's
    # arithmetic mean instead; resampling may still introduce interpolation
    # overshoot, so native-file peaks are measured separately for the report.
    average = "+".join(f"{1/channels:.17g}*c{i}" for i in range(channels))
    cmd += ["-vn", "-af", f"pan=mono|c0={average}", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    result = subprocess.run(cmd, check=True, capture_output=True)
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not samples.size:
        raise ValueError(f"Empty decoded audio: {path}")
    return samples


def probe(path: Path) -> dict:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                             "-show_entries", "stream=sample_rate,channels:format=duration",
                             "-of", "json", str(path)], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return {"duration_s": float(data["format"]["duration"]),
            "sample_rate": int(stream["sample_rate"]), "channels": int(stream["channels"])}


def db(value: float) -> float:
    return float(20 * math.log10(max(float(value), 1e-12)))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_file_levels(path: Path) -> dict:
    """Native decoded-file sample peak and arithmetic-mono RMS, no resampling."""
    import soundfile as sf

    y, sr = sf.read(path, always_2d=True)
    if not y.size or not np.isfinite(y).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    mono = y.mean(axis=1)
    rms, peak = float(np.sqrt(np.mean(mono**2))), float(np.max(np.abs(y)))
    return {"rms_dbfs": round(db(rms), 3), "sample_peak_dbfs": round(db(peak), 3),
            "crest_db": round(db(peak)-db(rms), 3),
            "level_measurement_basis": "native_file_sample_peak_all_channels_and_arithmetic_mono_rms_no_resampling",
            "native_sample_rate": sr}


def _frames(y, size, hop):
    if len(y) < size:
        y = np.pad(y, (0, size - len(y)))
    return np.lib.stride_tricks.sliding_window_view(y, size)[::hop]


def descriptors(y: np.ndarray, sr: int, tempo_bpm: float | None = None,
                beat_offset_s: float | None = None) -> dict:
    """Small deterministic measurements, with explicit limits of inference."""
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(y)))
    block = max(1, int(sr * .4))
    frames = _frames(y, block, block)
    levels = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    size, hop = 2048, 512
    frames = _frames(y, size, hop)
    spec = np.abs(np.fft.rfft(frames * np.hanning(size), axis=1))
    energy = spec ** 2
    freq = np.fft.rfftfreq(size, 1 / sr)
    total = max(float(energy.sum()), 1e-12)
    band_defs = {"low_30_160hz": (30, 160), "mid_160_2000hz": (160, 2000),
                 "high_2000_10000hz": (2000, min(10000, sr / 2))}
    shares, activity = {}, {}
    for name, (lo, hi) in band_defs.items():
        mask = (freq >= lo) & (freq < hi)
        band_energy = energy[:, mask].sum(axis=1)
        shares[name] = round(float(band_energy.sum()) / total, 5)
        # Relative-to-this-band peak, a descriptor of intermittent activity;
        # it cannot establish that a bass/drum/guitar was present.
        ceiling = float(np.max(band_energy))
        activity[name] = round(float(np.mean(band_energy > ceiling * .01)), 4) if ceiling > 1e-10 else 0.0
    flux = np.maximum(np.diff(spec, axis=0), 0).sum(axis=1)
    if flux.size and float(np.max(flux)) > 1e-9:
        smooth = np.convolve(flux, np.ones(9) / 9, mode="same")
        threshold = float(np.median(smooth) + .8 * np.std(smooth))
        candidates = np.flatnonzero((smooth[1:-1] > smooth[:-2]) &
                                   (smooth[1:-1] >= smooth[2:]) &
                                   (smooth[1:-1] > threshold)) + 1
        events = []
        for n in candidates:
            t = float((n + 1) * hop / sr)
            if not events or t - events[-1] >= .10:
                events.append(t)
    else:
        events = []
    estimated = None
    if len(flux) > 10 and float(np.max(flux)) > 1e-9:
        env = flux - np.mean(flux)
        nfft = 1 << (2 * len(env) - 1).bit_length()
        spectrum = np.fft.rfft(env, nfft)
        corr = np.fft.irfft(spectrum * np.conjugate(spectrum), nfft)[:len(env)]
        low, high = int(60 * sr / (180 * hop)), min(len(env) - 1, int(60 * sr / (60 * hop)))
        if high > low:
            lag = low + int(np.argmax(corr[low:high + 1]))
            confidence = float(corr[lag] / max(corr[0], 1e-12))
            if confidence > .05:
                estimated = {"bpm": round(60 * sr / (lag * hop), 2),
                             "periodicity": round(confidence, 4),
                             "status": "estimate_unverified_half_double_tempo_possible"}
    offbeat = None
    if tempo_bpm and beat_offset_s is not None and events:
        period = 60 / tempo_bpm
        phase = (np.asarray(events) - beat_offset_s) / period
        distance = np.abs(phase - np.round(phase))
        offbeat = round(float(np.mean(distance > .20)), 4)
    return {
        "duration_s": round(len(y) / sr, 4), "rms_dbfs": round(db(rms), 3),
        "sample_peak_dbfs": round(db(peak), 3), "crest_db": round(db(peak) - db(rms), 3),
        "rms_400ms_p90_minus_p10_db": round(db(np.percentile(levels, 90)) - db(np.percentile(levels, 10)), 3),
        "silence_400ms_share_below_minus45_dbfs": round(float(np.mean(levels < 10 ** (-45 / 20))), 4),
        "band_energy_share": shares, "relative_band_activity_share": activity,
        "mix_transient_events_per_s": round(len(events) / (len(y) / sr), 4),
        "beat_estimate": estimated, "offbeat_transient_share": offbeat,
        "rhythm_caveat": "Mix transients include consonants and all instruments; offbeat share is not a validated groove/syncopation score.",
        "instrument_identity": "unknown_from_mix",
    }


def validate_annotations(data: dict, duration: float) -> list[dict]:
    sections = data.get("sections", [])
    if not sections:
        raise ValueError("Annotations require nonempty sections")
    previous = 0.0
    for sec in sections:
        start, end = float(sec["start_s"]), float(sec["end_s"])
        if not sec.get("name") or not math.isfinite(start + end) or start < 0 or end <= start:
            raise ValueError(f"Invalid section: {sec}")
        if start < previous - 1e-6 or end > duration + .1:
            raise ValueError("Annotated sections overlap, are out of order, or exceed the audio")
        previous = end
    tempo = data.get("tempo_bpm")
    if tempo is not None and (not math.isfinite(float(tempo)) or not 20 <= float(tempo) <= 300):
        raise ValueError("tempo_bpm must be 20..300 or null")
    return sections


def blueprint(reference: Path, annotation_file: Path, stems: dict[str, str] | None = None) -> dict:
    annotations = json.loads(annotation_file.read_text(encoding="utf-8-sig"))
    audio = probe(reference)
    sections = validate_annotations(annotations, audio["duration_s"])
    sr = 22050
    y = decode(reference, sr)
    stem_audio = {}
    for role, path in (stems or {}).items():
        stem_path = Path(path)
        if not stem_path.is_absolute():
            stem_path = annotation_file.parent / stem_path
        samples = decode(stem_path, sr)
        if abs(len(samples) - len(y)) / sr > .15:
            raise ValueError(f"Stem {role} is not time-aligned to the reference duration")
        stem_audio[role] = (str(stem_path.resolve()), samples)
    rows = []
    for sec in sections:
        start, end = float(sec["start_s"]), float(sec["end_s"])
        tempo = sec.get("tempo_bpm", annotations.get("tempo_bpm"))
        offset = annotations.get("first_beat_s")
        relative_beat = None if offset is None else float(offset) - start
        values = descriptors(y[int(start * sr):int(end * sr)], sr, tempo, relative_beat)
        roles = {}
        for role, (path, samples) in stem_audio.items():
            segment = samples[int(start * sr):int(end * sr)]
            step = int(.4 * sr)
            window = _frames(segment, step, step)
            rms = np.sqrt(np.mean(window.astype(np.float64) ** 2, axis=1))
            roles[role] = {"source": "user_supplied_aligned_stem", "path": path,
                           "activity_share_above_minus45_dbfs": round(float(np.mean(rms > 10 ** (-45 / 20))), 4),
                           "rms_dbfs": round(db(np.sqrt(np.mean(segment.astype(np.float64) ** 2))), 3),
                           "caveat": "Stem bleed and separation errors can appear as activity."}
        manual = sec.get("roles", {})
        rows.append({"name": sec["name"], "start_s": start, "end_s": end,
                     "annotation_status": sec.get("status", annotations.get("status", "draft")),
                     "tempo_bpm_annotated": tempo,
                     "approx_bars_at_annotated_tempo": round((end-start) * tempo / 60 / annotations.get("beats_per_bar", 4), 3) if tempo else None,
                     "measured_mix": values, "measured_role_activity": roles,
                     "manual_role_annotations": manual,
                     "role_observation_status": "manual_and_stems" if manual or roles else "unknown_no_instrument_detector",
                     "target": dict(sec.get("target", {}))})
    measured_rms = [s["measured_mix"]["rms_dbfs"] for s in rows]
    median = float(np.median(measured_rms))
    for row in rows:
        # Defaults are explicitly proposals, not extracted instrument commands.
        row["target"].setdefault("ins_pattern", "full" if row["measured_mix"]["rms_dbfs"] >= median else "sparse")
        row["target"].setdefault("proposal_status", "hypothesis_requires_listening")
    return {"schema": 1, "workflow": "reference_arrangement_blueprint",
            "reference_audio": str(reference.resolve()), "reference_probe": audio,
            "annotations": str(annotation_file.resolve()),
            "annotation_status": annotations.get("status", "draft"), "sections": rows,
            "limits": ["Frequency bands do not identify drums, bass, strings or guitars.",
                       "Tempo estimates may be half/double time; annotate beats manually for rhythmic interpretation.",
                       "Ins controls a single symbolic voice; exact multi-track orchestration needs MIDI/DAW stems.",
                       "No proxy score establishes melody/hook identity, musical quality, or a high-scoring remake."]}


def loudness_match(paths: dict[str, Path], out: Path, target_dbfs=-20.0) -> dict:
    """One gain per file, one common safe RMS target; never peak-limit each arm."""
    import soundfile as sf

    measures = {}
    for label, path in paths.items():
        y, sr = sf.read(path, always_2d=True)
        if not y.size or not np.isfinite(y).all():
            raise ValueError(f"Empty or nonfinite audio: {path}")
        rms = float(np.sqrt(np.mean(y.mean(axis=1) ** 2)))
        peak = float(np.max(np.abs(y)))
        if rms <= 1e-8 or peak <= 1e-8:
            raise ValueError(f"Silent audio cannot be loudness matched: {path}")
        measures[label] = {"rms": rms, "peak": peak, "sr": sr,
                           "duration_s": len(y)/sr, "safe_rms": .98*rms/peak,
                           "source_sha256": _file_sha256(path)}
    common = min(10**(target_dbfs / 20), *(v["safe_rms"] for v in measures.values()))
    out.mkdir(parents=True, exist_ok=True)
    report = {"method": "linear_gain_common_mono_RMS", "requested_rms_dbfs": target_dbfs,
              "common_rms_dbfs": db(common),
              "caveat": "RMS matching is not BS.1770 perceptual LUFS matching. Check perceptual loudness by ear.", "files": {}}
    for label, path in paths.items():
        y, sr = sf.read(path, always_2d=True)
        gain = common / measures[label]["rms"]
        destination = out / f"{label}.wav"
        sf.write(destination, y * gain, sr, subtype="PCM_24")
        report["files"][label] = {"source": str(path.resolve()), "listening_file": str(destination.resolve()),
                                   "source_sha256": measures[label]["source_sha256"],
                                   "listening_sha256": _file_sha256(destination),
                                   "rms_in_dbfs": db(measures[label]["rms"]),
                                   "source_sample_peak_dbfs": db(measures[label]["peak"]),
                                   "gain_db": db(gain), "rms_out_dbfs": db(common),
                                   "sample_peak_out_dbfs": db(measures[label]["peak"] * gain),
                                   "duration_s": measures[label]["duration_s"]}
    return report
