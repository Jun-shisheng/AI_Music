"""Profile a *reference* track: how dense and how high does the singing get?

Why this exists (2026-10-16, docs/arrangement_spec.md §1.6): the user's method is
"imitate the great ones first, then create" -- and the property being imitated is
exactly the one prompt text cannot control: the contrast between a word-sparse
low verse and a word-dense high chorus. Measured on our own delivery, the
chorus/verse density ratio is 1.14 where the reference behaviour is 1.5-2.5x.
A live number for the target can only come from the reference recordings
themselves (human_refs/refs/), so this tool turns a reference track into the
same kind of number `plan_metrics.py` produces for our plans.

What it measures, per 10-second window, on the separated vocal:
  rms_db        how loud the section is (used to group windows)
  onsets_per_s  vocal articulations per second -- how fast the words come
  f0_median     median pitch of the sung voice in Hz (the register)

then reports the contrast between the track's dense (top-quartile RMS) and soft
(25th-60th percentile RMS) windows:
  density_ratio             how many times denser the loud parts are
  register_delta_semitones  how much higher they sit

The grouping is deliberately loudness-based rather than "verse/chorus": automatic
structural segmentation is fragile, and what we need is the *size of the
contrast*, not the labels. The intro/outro are excluded by the percentile band,
because a near-silent intro would otherwise read as an absurdly sparse verse.

Everything is cached under the work directory: re-running after a crash (or with
more tracks) only does the missing steps. Demucs is optional -- if the model
weights cannot be fetched (this machine cannot reach huggingface.co), the tool
falls back to measuring the full mix, which still carries the density contrast
(articulation is transient, so it survives the backing track) but not the pitch
curve, which the instrumental would contaminate.

Usage (tools/vocal_repair/.venv has demucs + librosa + pyworld; the YuE2/ACE
venvs do not):
  tools/vocal_repair/.venv/Scripts/python.exe -I tools/generation/ref_profile.py \
      --track human_refs/refs/hoyo-mix_wildfire.m4a --json logs/ref_wildfire.json
  tools/vocal_repair/.venv/Scripts/python.exe -I tools/generation/ref_profile.py \
      --all --report human_refs/refs/_profiles.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

WINDOW_S = 10.0
ANALYSIS_SR = 24000          # enough for density + F0, and 2x faster than 48k
# Bump when the *analysis* of the windows changes: the whisper pass (minutes per
# track) stays cached in its own file, and a summary-only change is recomputed
# from the cached windows without re-reading the audio.
#   v2: density percentiles restricted to windows that actually contain singing,
#       plus onsets_per_word.
#   v3: per-window chroma and hook_similarity -- a per-window measurement, so a
#       v2 cache has to re-read the stem (still cheap: the stem wav is cached).
SCHEMA = 3
ASR_PY = PROJECT_ROOT / "tools" / "asr" / ".venv" / "Scripts" / "python.exe"
WORD_RATE = Path(__file__).resolve().parent / "word_rate.py"


# --- cached steps -----------------------------------------------------------

def decode(src: Path, work: Path) -> Path:
    """m4a/mp3/... -> mono 24 kHz float wav (loudness analysis does not care)."""
    out = work / f"{src.stem}.wav"
    if out.exists():
        return out
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-ac", "1", "-ar", str(ANALYSIS_SR), str(out)],
        check=True)
    return out


def separate_vocals(wav: Path, work: Path) -> Path | None:
    """Run demucs (htdemucs) and return the vocal stem, or None if unavailable.

    CPU-only here -- the vocal_repair venv's torch has no CUDA -- so this is the
    slow step (~2-5 min per track) and every track's stem is cached.
    """
    out = work / f"{wav.stem}_vocals.wav"
    if out.exists():
        return out
    stem = work / "demucs" / "htdemucs" / wav.stem / "vocals.wav"
    if not stem.exists():
        try:
            subprocess.run(
                [sys.executable, "-m", "demucs", "--two-stems", "vocals",
                 "-o", str(work / "demucs"), str(wav)],
                check=True, capture_output=True, timeout=3600)
        except Exception as exc:                                  # noqa: BLE001
            print(f"    demucs unavailable ({type(exc).__name__}); "
                  f"measuring the full mix for density only", flush=True)
            return None
    if not stem.exists():
        return None
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(stem), "-ac", "1", "-ar", str(ANALYSIS_SR),
                    str(out)], check=True)
    return out


# --- measurements -----------------------------------------------------------

def windows_of(y: np.ndarray, sr: int) -> list[dict]:
    """Per-window loudness, articulation rate and F0 of one signal."""
    import librosa

    hop = int(WINDOW_S * sr)
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, units="frames", delta=0.05)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    except Exception:                                             # noqa: BLE001
        onset_times = np.array([])
    try:
        import pyworld
        x = y.astype(np.float64)
        f0, t = pyworld.dio(x, sr, frame_period=10.0)
        f0 = pyworld.stonemask(x, f0, t, sr)
    except Exception:                                             # noqa: BLE001
        f0, t = np.zeros(0), np.zeros(0)
    # Chroma per window: the harmonic content the loud windows can be compared
    # against each other for (see hook_similarity).
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
        chroma_times = librosa.times_like(chroma, sr=sr, hop_length=2048)
    except Exception:                                             # noqa: BLE001
        chroma, chroma_times = np.zeros((12, 0)), np.zeros(0)

    rows: list[dict] = []
    for start in range(0, len(y), hop):
        chunk = y[start:start + hop]
        if len(chunk) < sr:                    # ignore a <1 s tail stub
            continue
        t0, t1 = start / sr, (start + len(chunk)) / sr
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        onsets = int(np.sum((onset_times >= t0) & (onset_times < t1)))
        if len(f0):
            voiced = f0[(t >= t0) & (t < t1) & (f0 > 60)]
        else:
            voiced = np.zeros(0)
        row = {
            "t0": round(t0, 1),
            "rms_db": round(20 * np.log10(max(rms, 1e-9)), 1),
            "onsets_per_s": round(onsets / WINDOW_S, 2),
            "f0_median": round(float(np.median(voiced)), 1) if len(voiced) else None,
            "voiced_share": round(float(len(voiced)) / max(len(f0) / 100 * WINDOW_S, 1), 2)
            if len(f0) else None,
        }
        if chroma.shape[1]:
            sel = (chroma_times >= t0) & (chroma_times < t1)
            if sel.any():
                row["chroma"] = [round(float(v), 3) for v in chroma[:, sel].mean(axis=1)]
        rows.append(row)
    return rows


def hook_similarity(rows: list[dict]) -> float | None:
    """How much the loud windows sound like each other (chroma cosine).

    A chorus that is a repeated hook -- the same melody and chords every time it
    comes round -- makes its loud windows resemble one another; a through-composed
    song whose "chorus" is a new melody each time does not. Measured on the vocal
    stem's harmonic content so it can be compared across tracks that have no
    symbolic plan to inspect (the references).
    """
    vecs = [np.array(r["chroma"], dtype=float) for r in rows if r.get("chroma")]
    rms = [r["rms_db"] for r in rows if r.get("chroma")]
    if len(vecs) < 6:
        return None
    loud = [v for v, level in zip(vecs, rms)
            if level >= float(np.percentile(rms, 75))]
    if len(loud) < 3:
        return None
    sims: list[float] = []
    for i in range(len(loud)):
        for j in range(i + 1, len(loud)):
            a, b = loud[i], loud[j]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom:
                sims.append(float(np.dot(a, b) / denom))
    return round(float(np.mean(sims)), 3) if sims else None


def group_contrast(rows: list[dict]) -> dict:
    """How much denser and higher the song's loud parts are than its soft ones.

    Two views, because the mean of two bands understates what a listener hears:

      band means (soft = 25-60th percentile of RMS, loud = top quartile) say
      where the two groups sit;
      floor vs peak (the 25th vs the 90th percentile of the *density* itself)
      says how big the swing is, which is the number a chorus-vs-verse target
      needs. On Wildfire the band means read x1.2 while the swing is x1.8 --
      averaging across a song's many mid-density sections flattens the contrast.

    Percentiles rather than min/max so a near-silent intro or a fade-out cannot
    masquerade as a verse, and a single loud hit cannot masquerade as a chorus.
    """
    if not rows:
        return {}
    density = np.array([r["onsets_per_s"] for r in rows], dtype=float)
    rms = np.array([r["rms_db"] for r in rows])
    order = np.argsort(rms)
    n = len(rows)
    soft_idx = [order[i] for i in range(int(n * 0.25), max(int(n * 0.60), 1))]
    loud_idx = [order[i] for i in range(int(n * 0.75), n)]
    if not soft_idx or not loud_idx:
        return {}
    soft = [rows[i] for i in soft_idx]
    loud = [rows[i] for i in loud_idx]

    def mean(rows_: list[dict], key: str) -> float | None:
        vals = [r[key] for r in rows_ if r.get(key) is not None]
        return round(float(np.mean(vals)), 2) if vals else None

    soft_d, loud_d = mean(soft, "onsets_per_s"), mean(loud, "onsets_per_s")
    soft_f, loud_f = mean(soft, "f0_median"), mean(loud, "f0_median")
    floor_d = round(float(np.percentile(density, 25)), 2)
    peak_d = round(float(np.percentile(density, 90)), 2)
    # pyworld on a demucs stem sometimes tracks a low instrument instead of the
    # voice (Wildfire reads 136-170 Hz in two windows, an octave under its
    # 290-370 Hz speech range), so the register number is the weaker one and is
    # flagged by how much of the track was voiced at all.
    voiced = [r["voiced_share"] for r in rows if r.get("voiced_share") is not None]
    words = np.array([r["words_per_s"] for r in rows
                      if r.get("words_per_s") is not None], dtype=float)
    extra: dict = {}
    if len(words) >= 4:
        # Only windows where someone is actually singing. A plain percentile over
        # every window measures how long the instrumental intro is: Sabaton's
        # floor came out 0.10 and Skillet's 0.00 purely from silent-vocal windows,
        # which would have made every intro-heavy reference look like a x25 swing.
        singing = words[words > 0]
        w_floor = round(float(np.percentile(singing, 25)), 2) if len(singing) >= 4 else None
        w_peak = round(float(np.percentile(singing, 90)), 2) if len(singing) >= 4 else None
        extra = {
            "floor_words_per_s": w_floor,
            "peak_words_per_s": w_peak,
            "swing_words": (round(w_peak / w_floor, 2)
                            if w_floor and w_peak else None),
            # Share of the track with any singing at all: ~0 marks an instrumental
            # cue (the Two Steps From Hell material), which has no density target
            # to imitate.
            "singing_share": round(float(len(singing)) / len(words), 2),
            # Onset-detector hits per sung word: ~1 means the onsets are the
            # voice, >>1 means percussion bleed is being counted (whisper_words).
            "onsets_per_word": round(float(np.mean(density)) / float(np.mean(words)), 2)
            if float(np.mean(words)) else None,
        }
    return extra | {
        "hook_similarity": hook_similarity(rows),
        "windows": n,
        "soft_rms_db": mean(soft, "rms_db"),
        "loud_rms_db": mean(loud, "rms_db"),
        "soft_onsets_per_s": soft_d,
        "loud_onsets_per_s": loud_d,
        "floor_onsets_per_s": floor_d,
        "peak_onsets_per_s": peak_d,
        "swing_ratio": round(peak_d / floor_d, 2) if floor_d else None,
        "soft_f0_hz": soft_f,
        "loud_f0_hz": loud_f,
        "voiced_share_median": round(float(np.median(voiced)), 2) if voiced else None,
        "density_ratio": round(loud_d / soft_d, 2) if soft_d and loud_d else None,
        "register_delta_semitones":
            round(12 * float(np.log2(loud_f / soft_f)), 1)
            if soft_f and loud_f else None,
    }


def whisper_words(target: Path, work: Path) -> dict | None:
    """Sung words/s per window from the ASR runtime, cached.

    The onset detector above counts *every* transient in the stem, and a demucs
    vocal stem of a percussion-heavy battle mix still carries drum bleed: our own
    01_hammer_and_hymn reads 2.9-4.8 onsets/s while whisper counts 1.16-1.27
    words/s in the same 10 s windows, i.e. 2.4-4 onsets per word. So onsets are a
    fast, biased screen and this is the number to trust for a target -- it is
    also the number `plan_metrics.py` was validated against on our own delivery
    (words/s vs plan onsets/s tracked section by section, ratio 0.74-0.93).

    Runs `word_rate.py` under tools/asr/.venv (faster-whisper lives there, not in
    this venv). Its windows step by the same WINDOW_S from 0 as the onset curve,
    so the two line up index by index.
    """
    out = work / f"{target.stem}_words.json"
    if not out.exists():
        if not ASR_PY.exists():
            print("    no ASR runtime (tools/asr/.venv); skipping words", flush=True)
            return None
        try:
            subprocess.run(
                [str(ASR_PY), "-I", str(WORD_RATE), "--wav", str(target),
                 "--window", str(WINDOW_S), "--language", "auto",
                 "--json", str(out)],
                check=True, capture_output=True, timeout=7200)
        except Exception as exc:                                  # noqa: BLE001
            print(f"    whisper failed ({type(exc).__name__}); onsets only",
                  flush=True)
            return None
    try:
        return json.loads(out.read_text(encoding="utf-8"))
    except Exception:                                             # noqa: BLE001
        return None


def merge_words(rows: list[dict], words: dict | None) -> None:
    """Attach words/s to each onset window (in place), same window grid."""
    if not words:
        return
    curve = {round(c["t0"], 1): c for c in words.get("curve", [])}
    for row in rows:
        hit = curve.get(row["t0"])
        if hit:
            row["words"] = hit["words"]
            row["words_per_s"] = hit["words_per_s"]
    if words.get("lang"):
        rows[0]["lang"] = words["lang"]


def profile_one(src: Path, work: Path, separate: bool = True,
                whisper: bool = False) -> dict:
    """Profile one track, caching the result in the work dir.

    The cache is written per track, not only at the end of a run: profiling the
    reference set is a ~1 hour CPU job (demucs on CPU), and a crash at track 11
    must not throw away tracks 1-10. Re-running reloads a finished track instead
    of recomputing it.
    """
    cached = work / f"{src.stem}_profile.json"
    if cached.exists():
        try:
            data = json.loads(cached.read_text(encoding="utf-8"))
            # A cached onset-only profile is not a cached whisper profile.
            if not (whisper and not data.get("whisper")):
                if data.get("schema") == SCHEMA:
                    print(f"\n== {src.name} (cached)", flush=True)
                    return data
                # Older schema. If the windows already carry what the current
                # summary needs, only the summary changed -- re-derive it instead
                # of redoing work. Otherwise fall through and re-measure (the
                # whisper pass is still cached on disk, so it is not repeated).
                windows = data.get("windows") or []
                if windows and windows[0].get("chroma") is not None:
                    print(f"\n== {src.name} (cached windows, "
                          f"resummarising v{data.get('schema')} -> v{SCHEMA})",
                          flush=True)
                    data["schema"] = SCHEMA
                    data["contrast"] = group_contrast(windows)
                    cached.write_text(
                        json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8")
                    return data
                print(f"\n== {src.name} (re-measuring for v{SCHEMA}; "
                      f"whisper stays cached)", flush=True)
        except Exception:                                         # noqa: BLE001
            pass

    print(f"\n== {src.name}", flush=True)
    wav = decode(src, work)
    vocal = separate_vocals(wav, work) if separate else None
    target = vocal or wav
    print(f"    source={'vocal stem' if vocal else 'full mix'} "
          f"({target.stat().st_size / 1e6:.1f} MB)", flush=True)
    import soundfile as sf
    y, sr = sf.read(target, always_2d=False)
    rows = windows_of(np.asarray(y, dtype=np.float32), sr)
    if whisper:
        merge_words(rows, whisper_words(target, work))
    data = {
        "file": src.name,
        "title": " ".join(src.stem.split("_")[1:]).replace("-", " "),
        "duration_s": round(len(y) / sr, 1),
        "isolated_vocal": bool(vocal),
        "whisper": whisper,
        "schema": SCHEMA,
        "windows": rows,
        "contrast": group_contrast(rows),
    }
    cached.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    return data


def print_profile(p: dict) -> None:
    c = p.get("contrast") or {}
    print(f"    duration {p['duration_s']}s | vocal "
          f"{'isolated' if p['isolated_vocal'] else 'MIX (pitch unreliable)'}")
    if not c:
        print("    not enough windows to compare")
        return
    print(f"    soft  windows: {c['soft_onsets_per_s']:.2f} onsets/s  "
          f"F0 {c['soft_f0_hz']} Hz  ({c['soft_rms_db']} dBFS)")
    print(f"    loud  windows: {c['loud_onsets_per_s']:.2f} onsets/s  "
          f"F0 {c['loud_f0_hz']} Hz  ({c['loud_rms_db']} dBFS)")
    print(f"    -> floor {c['floor_onsets_per_s']:.2f} -> peak "
          f"{c['peak_onsets_per_s']:.2f} onsets/s  = swing x{c['swing_ratio']}  "
          f"(band means x{c['density_ratio']}), "
          f"register {c['register_delta_semitones']:+} semitones")
    if c.get("floor_words_per_s") is not None:
        print(f"    -> words: floor {c['floor_words_per_s']:.2f} -> peak "
              f"{c['peak_words_per_s']:.2f} word/s = swing x{c['swing_words']}"
              f"   [{c['onsets_per_word']} onsets per word]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=Path, action="append", default=[],
                    help="profile this file; repeatable. Explicit tracks are "
                         "profiled before --all's, so a single invocation can do "
                         "\"our own take first, then the whole reference set\"")
    ap.add_argument("--all", action="store_true",
                    help="profile every audio file in --refs-dir")
    ap.add_argument("--refs-dir", type=Path,
                    default=PROJECT_ROOT / "human_refs" / "refs")
    ap.add_argument("--work-dir", type=Path,
                    default=PROJECT_ROOT / "human_refs" / "refs" / "_work")
    ap.add_argument("--no-separate", action="store_true",
                    help="skip demucs (faster; pitch curve becomes unreliable)")
    ap.add_argument("--whisper", action="store_true",
                    help="also transcribe with whisper for a words/s curve -- "
                         "the trustworthy density number (onsets count drum "
                         "bleed); ~2-3 min per track on CPU")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None,
                    help="markdown table across all profiled tracks")
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    tracks = list(args.track)
    if args.all:
        tracks += sorted(p for p in args.refs_dir.iterdir()
                         if p.suffix.lower() in (".m4a", ".mp3", ".flac", ".wav", ".opus"))
    if not tracks:
        ap.error("give --track FILE (repeatable) and/or --all")
    if not any(p.exists() for p in tracks):
        print(f"no audio found in {args.refs_dir}")
        return 1
    tracks = [p for p in tracks if p.exists()]

    profiles = [profile_one(t, args.work_dir, not args.no_separate, args.whisper)
                for t in tracks]
    for p in profiles:
        print_profile(p)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(profiles, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"\njson -> {args.json}")

    if args.report:
        rows = [(p["file"], p["duration_s"], p["isolated_vocal"], p["contrast"])
                for p in profiles]
        lines = ["# 参考曲剖面（ref_profile.py）", "",
                 f"- 窗口 {WINDOW_S:g}s；soft = RMS 25–60 百分位，loud = 顶部 25%",
                 "- **words 列是可信口径**（whisper，每 10 秒窗）；onsets 列是快速筛选，",
                 "  会被鼓串进人声轨的瞬态污染，`ons/词` 就是污染比（越接近 1 越干净）。",
                 "- `floor→peak` = 该速率自身的 25 → 90 百分位（**这是要模仿的摆幅**）",
                 "- `band x` = loud/soft 均值比（会被曲中大量中密度段拉平，仅供参考）",
                 "- 纯器乐曲（人声 floor ≈ 0）不参与平均，已单列标记",
                 "", "| 曲目 | 时长 | floor 词/s | peak 词/s | **swing(词)** | ons/词 "
                 "| 演唱窗 | floor 起音/s | peak 起音/s | band x | **hook 相似度** | register |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        vocal_rows, instr_rows = [], []
        for name, dur, iso, c in rows:
            if not c:
                lines.append(f"| {name} | {dur}s | – | – | – | – | – | – | – | – | – | – |")
                continue
            w_floor = c.get("floor_words_per_s")
            share = c.get("singing_share")
            is_instrumental = share is not None and share < 0.5
            words_cell = ("–" if w_floor is None else
                          f"{w_floor:.2f} | {c['peak_words_per_s']:.2f} | "
                          f"**x{c['swing_words']}** | {c['onsets_per_word']}")
            hook = c.get("hook_similarity")
            mark = " (器乐?)" if is_instrumental else ""
            lines.append(
                f"| {name}{mark} | {dur}s | {words_cell} | "
                f"{'' if share is None else f'{share * 100:.0f}%'} | "
                f"{c['floor_onsets_per_s']:.2f} | {c['peak_onsets_per_s']:.2f} | "
                f"x{c['density_ratio']} | "
                f"{'–' if hook is None else f'{hook:.3f}'} | "
                f"{c['register_delta_semitones']:+.1f} 半音 |")
            (instr_rows if is_instrumental else vocal_rows).append(c)

        def med(rows_: list[dict], key: str) -> float | None:
            vals = [r[key] for r in rows_ if r.get(key) is not None]
            return float(np.median(vals)) if vals else None

        if vocal_rows:
            lines += ["", f"### 人声曲合计（n={len(vocal_rows)}"
                          f"{f'，另 {len(instr_rows)} 首疑为器乐已排除' if instr_rows else ''}）"]
            for label, key, fmt in (
                    ("swing（词）", "swing_words", "x{:.2f}"),
                    ("floor 词/s", "floor_words_per_s", "{:.2f}"),
                    ("peak 词/s", "peak_words_per_s", "{:.2f}"),
                    ("band x", "density_ratio", "x{:.2f}"),
                    ("音区抬升", "register_delta_semitones", "{:+.1f} 半音"),
                    ("ons/词（污染比）", "onsets_per_word", "{:.2f}"),
                    ("hook 相似度", "hook_similarity", "{:.3f}")):
                value = med(vocal_rows, key)
                if value is None:
                    continue
                lo = float(np.percentile([r[key] for r in vocal_rows
                                          if r.get(key) is not None], 25))
                hi = float(np.percentile([r[key] for r in vocal_rows
                                          if r.get(key) is not None], 75))
                lines.append(f"- **{label}**：中位 " + fmt.format(value)
                             + "   （四分位 " + fmt.format(lo) + " – "
                             + fmt.format(hi) + "）")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
