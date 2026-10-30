"""Time-budgeted YuE2 run over the epic-English battle briefs.

This is the wrapper around `yue2_batch_battle.py` that the 2026-10-14 session
needs, and it exists for three reasons:

1. **One process per title.** The CUDA OOM that killed six titles during the
   first YuE2 pass is cumulative -- the long-lived process ends up holding a
   fragmented allocator and the NAR prefill, which needs ~2.4 GiB in a single
   allocation, eventually fails.  A fresh process per title costs one model load
   (~1-2 min) and removes the failure mode entirely.

2. **A wall-clock budget.** The user asked for roughly six hours of GPU work.
   The loop checks the clock between titles and stops, so however long the model
   decides each song should be, the run ends when it should.  Every title is
   independent, so stopping early loses nothing.

3. **A duration gate.** YuE2 samples the length of its symbolic plan, so the
   same brief can come back 1:35 or 4:10.  The user's floor is 2:00 and the
   ceiling 3:30.  A promoted take outside that window is discarded and the title
   is re-run with a different seed base (the plan length is seed-dependent), up
   to `--length-retries` times.

State goes to `<out-dir>/RUN_STATUS.txt` so an interrupted session can be picked
up by reading a file instead of a chat log, and reruns skip promoted titles.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def needs_rerun(duration_s: float, min_s: float, max_s: float) -> str | None:
    """Reason to discard a promoted take, or None if its length is acceptable."""
    if duration_s < min_s:
        return f"too short ({duration_s:.0f}s < {min_s:.0f}s)"
    if duration_s > max_s:
        return f"too long ({duration_s:.0f}s > {max_s:.0f}s)"
    return None


def read_duration(wav: Path) -> float:
    import soundfile as sf
    return float(sf.info(str(wav)).duration)


def onset_reason(onset_s: float | None, max_s: float) -> str | None:
    """Reason to discard a promoted take for a late vocal entry, or None.

    The user's wording is "要么前奏几秒直接开唱，要么前奏时间 10-20 秒不能太长",
    so the ceiling is generous (20 s) and the plan trim is what actually gets the
    vocal in at a few seconds. None onset = the diction check did not run.
    """
    if onset_s is None or onset_s < 0:
        return None
    if onset_s > max_s:
        return f"vocal starts at {onset_s:.0f}s (> {max_s:.0f}s)"
    return None


def read_first_vocal(out_dir: Path, slug: str) -> float | None:
    """First-sung-word time of a promoted take, from the batch's cumulative index.

    Added 2026-10-15: the user's "前奏几秒直接开唱，要么前奏 10-20 秒不能太长".
    Measured on the 2026-10-14 delivery it was 15.5-32.3 s; the plan trim in
    `yue2_plan_edit` is the fix, and this is the gate that proves it worked.
    None when the diction check did not run (no ASR venv), in which case the
    gate is skipped rather than failing the title.
    """
    index = out_dir / "index.json"
    if not index.exists():
        return None
    try:
        rows = json.loads(index.read_text(encoding="utf-8"))
    except Exception:
        return None
    for row in rows:
        if row.get("title") == slug:
            value = row.get("first_vocal_s")
            return None if value is None or value < 0 else float(value)
    return None


def title_slugs(inputs: Path) -> list[str]:
    return sorted(p.name[: -len("_lyrics.txt")]
                  for p in inputs.glob("*_lyrics.txt")
                  if (inputs / p.name.replace("_lyrics.txt", "_tags.txt")).exists())


def select_slugs(slugs: list[str], filters: list[str] | None) -> list[str]:
    """Restrict a run to titles matching any filter substring.

    Needed to top up a batch without re-touching what it already delivered: a
    resumed run walks every brief and re-checks each promoted take against the
    duration gate, so without a filter a tightened ceiling silently re-rolls
    titles that are already published.
    """
    if not filters:
        return list(slugs)
    return [s for s in slugs if any(f in s for f in filters)]


def write_status(path: Path, **fields) -> None:
    lines = [f"{k}: {v}" for k, v in fields.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path,
                    default=PROJECT_ROOT / "inputs" / "2026-10-14_epic_en")
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "outputs" / "2026-10-14_epic_en")
    ap.add_argument("--python", type=Path,
                    default=PROJECT_ROOT / "vendor" / "YuE2" / ".venv" / "Scripts" / "python.exe")
    ap.add_argument("--script", type=Path,
                    default=PROJECT_ROOT / "tools" / "generation" / "yue2_batch_battle.py")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--candidates", type=int, default=2)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--nar-query-chunk", type=int, default=1024)
    ap.add_argument("--budget", type=float, default=12.0)
    ap.add_argument("--backend", default="torch-eager")
    ap.add_argument("--language", default="en")
    # The user's window: a 2:00 floor and a 3:30 target. The ceiling was a hard
    # 210 s after a 4:29 take on 2026-10-14 ("四分钟还是太长了"), but on
    # 2026-10-16 the user relaxed it -- "略微超过以后没啥问题" -- so a take a
    # little over 3:30 is now kept instead of costing a ~27 min re-roll (07's
    # 224 s take was discarded ten minutes before that message arrived).
    ap.add_argument("--min-duration", type=float, default=120.0)
    ap.add_argument("--max-duration", type=float, default=225.0)
    ap.add_argument("--length-retries", type=int, default=1,
                    help="extra rounds allowed per title when the take's length is "
                         "out of range, or when nothing was promoted at all")
    # 2026-10-15: the user wants the vocal in within a few seconds, or an intro
    # of at most 10-20 s. `--intro-bars` is the fix; this gate is the check.
    ap.add_argument("--max-onset", type=float, default=20.0,
                    help="discard a promoted take whose first sung word lands "
                         "later than this many seconds (needs the diction check)")
    ap.add_argument("--intro-bars", type=int, default=2,
                    help="trim the plan's instrumental intro to this many bars; "
                         "0 disables the plan edit entirely. Two bars is ~3 s of "
                         "count-in, which is what the references do (Arknights "
                         "Radiant sings at 0:02, HSR Wildfire at 0:03)")
    ap.add_argument("--shape", action="append", default=[],
                    help="plan_shape.py operator passed through to every title, "
                         "e.g. --shape 'split chorus:2' (repeatable). This is the "
                         "2026-10-16 density fix: the whole song used to run at one "
                         "rhythm, with the chorus measuring 1.14x the verse.")
    ap.add_argument("--title-timeout-min", type=float, default=55.0)
    ap.add_argument("--seed-base", type=int, default=1200)
    ap.add_argument("--slug-filter", nargs="*", default=None,
                    help="only titles whose slug contains one of these "
                         "substrings, e.g. --slug-filter 10_ 11_ 12_ to top up "
                         "a batch without re-checking what it already delivered")
    args = ap.parse_args()

    slugs = select_slugs(title_slugs(args.inputs), args.slug_filter)
    if not slugs:
        print(f"no briefs in {args.inputs}")
        return 1

    sel_dir = args.out_dir / "selected"
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.out_dir / "RUN_STATUS.txt"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    deadline = started + args.hours * 3600
    stamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"[{time.strftime('%H:%M:%S')}] {len(slugs)} titles, budget {args.hours} h, "
          f"candidates={args.candidates}, retries={args.retries}", flush=True)

    done: list[str] = []
    for slug in slugs:
        if time.time() >= deadline:
            print(f"[{time.strftime('%H:%M:%S')}] budget exhausted before {slug}",
                  flush=True)
            break
        remaining_h = (deadline - time.time()) / 3600
        round_no = 0          # completed subprocess rounds for this title
        while True:
            if time.time() >= deadline:
                print(f"[{time.strftime('%H:%M:%S')}] budget exhausted during {slug}",
                      flush=True)
                break

            remaining_h = (deadline - time.time()) / 3600
            seed_base = args.seed_base + round_no * 511
            cmd = [
                str(args.python), "-I", "-u", str(args.script),
                "--inputs", str(args.inputs), "--out-dir", str(args.out_dir),
                "--candidates", str(args.candidates), "--retries", str(args.retries),
                "--budget", str(args.budget), "--nar-query-chunk", str(args.nar_query_chunk),
                "--backend", args.backend, "--language", args.language,
                "--seed-base", str(seed_base), "--only", slug,
            ]
            if args.intro_bars:
                cmd += ["--intro-bars", str(args.intro_bars)]
            for op in args.shape:
                cmd += ["--shape", op]
            title_log = log_dir / f"epic_{slug}_{stamp}.log"
            print(f"[{time.strftime('%H:%M:%S')}] {slug}: start (seed base {seed_base}, "
                  f"{remaining_h:.1f} h left) -> {title_log.name}", flush=True)
            write_status(status_path, started=time.strftime("%Y-%m-%d %H:%M:%S",
                                                            time.localtime(started)),
                         deadline=time.strftime("%Y-%m-%d %H:%M:%S",
                                                time.localtime(deadline)),
                         current=slug, round=round_no + 1,
                         done=f"{len(done)}/{len(slugs)}", done_titles=", ".join(done))
            t0 = time.time()
            with open(title_log, "w", encoding="utf-8", errors="replace") as fh:
                try:
                    subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                   cwd=str(PROJECT_ROOT),
                                   timeout=args.title_timeout_min * 60, check=False)
                    rc: int | str = 0
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            print(f"[{time.strftime('%H:%M:%S')}] {slug}: process done in "
                  f"{(time.time() - t0) / 60:.1f} min (rc={rc})", flush=True)
            round_no += 1

            promoted = sel_dir / f"{slug}.wav"
            if not promoted.exists():
                if round_no > args.length_retries:
                    print(f"[{time.strftime('%H:%M:%S')}] {slug}: nothing promoted "
                          f"after {round_no} rounds; moving on", flush=True)
                    break
                print(f"[{time.strftime('%H:%M:%S')}] {slug}: nothing promoted; "
                      f"retrying in a fresh process (round {round_no + 1})", flush=True)
                continue

            try:
                dur = read_duration(promoted)
            except Exception as exc:
                print(f"  cannot read {promoted}: {exc}", flush=True)
                dur = 0.0
            reason = needs_rerun(dur, args.min_duration, args.max_duration)
            onset = read_first_vocal(args.out_dir, slug)
            if reason is None:
                reason = onset_reason(onset, args.max_onset)
            if reason is None and onset is not None:
                print(f"  vocal in at {onset:.1f}s", flush=True)
            if reason is None:
                print(f"[{time.strftime('%H:%M:%S')}] {slug} ok at {dur:.0f}s", flush=True)
                done.append(slug)
                break
            if round_no > args.length_retries:
                print(f"[{time.strftime('%H:%M:%S')}] {slug} kept at {dur:.0f}s "
                      f"({reason}) -- length retries exhausted", flush=True)
                done.append(slug)
                break
            print(f"[{time.strftime('%H:%M:%S')}] {slug} {reason}; discarding and "
                  f"re-running with a new seed (round {round_no + 1})", flush=True)
            for stale in (promoted, sel_dir / f"{slug}_raw.wav"):
                stale.unlink(missing_ok=True)

    elapsed = (time.time() - started) / 60
    write_status(status_path, started=time.strftime("%Y-%m-%d %H:%M:%S",
                                                    time.localtime(started)),
                 finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                 elapsed_min=f"{elapsed:.1f}", current="-",
                 done=f"{len(done)}/{len(slugs)}", done_titles=", ".join(done))
    print(f"[{time.strftime('%H:%M:%S')}] run finished: {len(done)}/{len(slugs)} titles "
          f"in {elapsed:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
