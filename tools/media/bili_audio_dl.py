#!/usr/bin/env python
"""Silent Bilibili audio downloader (CLI, no GUI) for reference tracks.

JiJiDown is a Tauri GUI app with no headless mode, so this talks to the same
Bilibili web API it uses, pulls the DASH audio-only stream, and remuxes it to
.m4a with ffmpeg. Nothing is displayed anywhere; everything goes to stdout and
the log file, per the repo rule "silent background, never pop a window".

Usage:
    python bili_audio_dl.py --out D:\\Model\\AI_Music\\human_refs\\refs \\
        --query "野火:Hoyo-Mix Wildfire" --query "耀斑:Hoyo-Mix Flares"

    python bili_audio_dl.py --out <dir> --bv BV1xx411c7mD --name "wildfire"

    # inspect candidates without downloading
    python bili_audio_dl.py --query "..." --list
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
COOKIE_FILE = os.path.join(os.environ.get("TEMP", "."), "bili_dl_cookie.txt")
FFMPEG = os.path.join(
    os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Links", "ffmpeg.exe"
)


def log(msg: str) -> None:
    print(msg, flush=True)


def http_json(url: str, referer: str = "https://www.bilibili.com") -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Referer": referer,
            "Accept": "application/json, text/plain, */*",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def prime_cookies() -> None:
    """Hit the homepage once so the anti-bot buvid3 cookie is set.

    The cookie jar must be installed globally, otherwise the buvid3 that the
    search endpoint requires is dropped and the request gets bounced to the
    anti-bot page.
    """
    urllib.request.install_opener(
        urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    )
    req = urllib.request.Request(
        "https://www.bilibili.com/", headers={"User-Agent": UA}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read(2000)
    except OSError as exc:  # pragma: no cover - network dependent
        log(f"  ! cookie prime failed: {exc}")


_EM = re.compile(r"</?em[^>]*>")


def search_videos(keyword: str, pages: int = 1) -> list[dict]:
    out: list[dict] = []
    for page in range(1, pages + 1):
        url = (
            "https://api.bilibili.com/x/web-interface/search/all/v2?keyword="
            + urllib.parse.quote(keyword)
            + f"&page={page}"
        )
        data = http_json(url).get("data") or {}
        for block in data.get("result") or []:
            if block.get("result_type") != "video":
                continue
            for item in block.get("data") or []:
                item["title_clean"] = _EM.sub("", item.get("title", ""))
                out.append(item)
        time.sleep(0.4)
    return out


def duration_s(text: str) -> int:
    parts = [int(p) for p in str(text).split(":") if p.isdigit()]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def pick_candidate(
    cands: list[dict],
    want_s: tuple[int, int] = (90, 450),
    require_groups: list[list[str]] | None = None,
    exclude: list[str] | None = None,
) -> tuple[dict | None, bool]:
    """Pick the best candidate. Returns (candidate, weak_match).

    Ranking on play count alone is not enough: Bilibili happily returns
    unrelated mega-popular videos for a loose query, and they outrank the real
    track. So `require_groups` (each group = alternatives, all groups must
    match) gates on the title, and only then does play count rank what is left.
    A weak match is one that had to fall back to the ungated ranking.
    """
    require_groups = require_groups or []
    exclude = [e.lower() for e in (exclude or [])]

    def eligible(c: dict, gated: bool) -> bool:
        dur = duration_s(c.get("duration", "0:00"))
        if not (want_s[0] <= dur <= want_s[1]):
            return False
        low = c["title_clean"].lower()
        if any(bad in low for bad in exclude):
            return False
        if gated:
            return all(
                any(alt.lower() in low for alt in group) for group in require_groups
            )
        return True

    def rank(c: dict) -> float:
        dur = duration_s(c.get("duration", "0:00"))
        score = float(int(c.get("play") or 0))
        low = c["title_clean"].lower()
        if any(k in low for k in ("cover", "翻唱", "钢琴", "教程", "reaction", "合集")):
            score *= 0.15
        if any(k in low for k in ("official", "官方", "mv", "ost", "原声", "纯音乐")):
            score *= 2.0
        if 100 <= dur <= 330:
            score *= 1.5
        return score

    pool = [c for c in cands if c.get("bvid")]
    for gated in (True, False):
        hits = [c for c in pool if eligible(c, gated)]
        if hits:
            hits.sort(key=rank, reverse=True)
            return hits[0], not gated
    return None, False


def resolve_cid(bvid: str) -> tuple[int, int]:
    data = http_json(
        f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    )["data"]
    return data["cid"], data.get("pages", [{}])[0].get("cid", data["cid"])


def audio_stream(bvid: str, cid: int) -> dict:
    url = (
        "https://api.bilibili.com/x/player/playurl"
        f"?bvid={bvid}&cid={cid}&fnval=16&fnver=0&fourk=1"
    )
    data = http_json(url)["data"]
    dash = data.get("dash") or {}
    audios = dash.get("audio") or []
    if not audios:
        raise RuntimeError("no DASH audio stream (video may be paid/region-locked)")
    audios.sort(key=lambda a: a.get("bandwidth", 0), reverse=True)
    return audios[0]


def download(url: str, dest: str, referer: str) -> int:
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Referer": referer}
    )
    tmp = dest + ".part"
    total = 0
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 18)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
    os.replace(tmp, dest)
    return total


def to_m4a(src: str, dest: str) -> bool:
    exe = FFMPEG if os.path.isfile(FFMPEG) else "ffmpeg"
    cmd = [
        exe, "-hide_banner", "-loglevel", "error", "-y",
        "-i", src, "-vn", "-c:a", "copy", "-movflags", "+faststart", dest,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"  ! ffmpeg failed: {exc}")
        return False
    if proc.returncode != 0:
        log(f"  ! ffmpeg rc={proc.returncode} {proc.stderr.strip()[:200]}")
        return False
    return True


def safe_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    return re.sub(r"\s+", " ", name)[:80]


def fetch_one(name: str, out_dir: str, query: str | None, bvid: str | None,
              dry: bool,
              require_groups: list[list[str]] | None = None,
              exclude: list[str] | None = None) -> dict:
    result = {"name": name, "status": "pending"}
    try:
        if not bvid:
            cands = search_videos(query)
            best, weak = pick_candidate(cands, require_groups=require_groups,
                                        exclude=exclude)
            if not best:
                result["status"] = "no-candidate"
                log(f"[{name}] no usable candidate for query {query!r}")
                return result
            bvid = best["bvid"]
            result["title"] = best["title_clean"]
            result["author"] = best.get("author")
            result["play"] = best.get("play")
            result["weak_match"] = weak
            log(
                f"[{name}] -> {bvid} | {best['title_clean']} | "
                f"UP={best.get('author')} | {best.get('duration')} | "
                f"play={best.get('play')}" + ("  [WEAK MATCH]" if weak else "")
            )
        if dry:
            result["status"] = "listed"
            return result

        cid, _ = resolve_cid(bvid)
        stream = audio_stream(bvid, cid)
        log(f"[{name}] audio {stream.get('bandwidth', 0)//1000} kbps "
            f"{stream.get('codecs')}")
        dest = os.path.join(out_dir, safe_name(name) + ".m4a")
        url = stream.get("baseUrl") or (stream.get("backupUrl") or [None])[0]
        referer = f"https://www.bilibili.com/video/{bvid}"
        # Bilibili serves a fragmented MP4 (.m4s). Remux it to a plain m4a so
        # downstream tools (Demucs, librosa, ffprobe) read it without fuss.
        raw = dest + ".m4s"
        size = download(url, raw, referer)
        if to_m4a(raw, dest):
            os.remove(raw)
        else:
            os.replace(raw, dest)
            log(f"[{name}] remux skipped, kept raw DASH stream")
        result.update({"status": "ok", "file": dest, "bytes": size, "bvid": bvid,
                       "kbps": stream.get("bandwidth", 0) // 1000})
        log(f"[{name}] saved {dest} ({size/1048576:.1f} MB)")
    except Exception as exc:  # noqa: BLE001 - report and continue the batch
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        log(f"[{name}] FAILED: {result['error']}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--query", action="append", default=[],
                    help='NAME:search terms[|req1;req2][|excl1,excl2]  '
                         '(repeatable; a req group is comma-separated '
                         'alternatives, all groups must appear in the title)')
    ap.add_argument("--bv", action="append", default=[],
                    help="NAME:BVid  (repeatable, skips search)")
    ap.add_argument("--list", action="store_true", dest="dry",
                    help="only resolve candidates, do not download")
    ap.add_argument("--manifest", default=None, help="write JSON results here")
    args = ap.parse_args()

    prime_cookies()
    os.makedirs(args.out, exist_ok=True)

    jobs: list[dict] = []
    for spec in args.query:
        parts = spec.split("|")
        name, _, q = parts[0].partition(":")
        groups = [
            [alt.strip() for alt in grp.split(",") if alt.strip()]
            for grp in (parts[1].split(";") if len(parts) > 1 else [])
            if grp.strip()
        ]
        excl = [e.strip() for e in (parts[2].split(",") if len(parts) > 2 else [])
                if e.strip()]
        jobs.append({
            "name": name.strip(),
            "query": q.strip() or name.strip(),
            "bvid": None,
            "require_groups": groups,
            "exclude": excl,
        })
    for spec in args.bv:
        name, _, bv = spec.partition(":")
        jobs.append({"name": name.strip(), "query": None, "bvid": bv.strip(),
                     "require_groups": [], "exclude": []})

    if args.dry:
        for job in jobs:
            if job["query"]:
                log(f"--- {job['name']}: {job['query']} "
                    f"req={job['require_groups']} excl={job['exclude']}")
                for c in search_videos(job["query"])[:8]:
                    log(f"    {c.get('bvid')} | {c.get('duration')} | "
                        f"{c.get('play')} | {c['title_clean'][:70]}")
        return 0

    results = [
        fetch_one(j["name"], args.out, j["query"], j["bvid"], args.dry,
                  j["require_groups"], j["exclude"])
        for j in jobs
    ]

    ok = sum(1 for r in results if r["status"] == "ok")
    log(f"--- {ok}/{len(results)} ok ---")
    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
