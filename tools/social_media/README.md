# Social Media Publishing Workflow

> **⚠️ 已废弃（2026-10-16 归档确认）**：2026-07 的"竖版短视频 + 发布"工作流，属 ACE-Step 时代，
> 与当前 YuE2 交付流程无关。本目录里 7 个一次性的批量造词脚本已移出到
> `_archive/deleted_20261016/social_media/`；下面记录的
> `curation.py` / `render_video.py` / `package_candidates.py` / `top_song_workflow.py` /
> `short_video.py` / `lyric_gate.py` 暂留（互相引用），确认不使用后可一并归档。

This directory is intentionally separate from the existing generation app. It reads the local SQLite library and `outputs/` audio files, then produces publish candidates and vertical lyric videos without changing the music generation pipeline.

## 1. Score The Library

```powershell
python tools/social_media/curation.py --limit 100
```

Outputs:

- `data/curation/curation_report_*.json`
- `data/curation/curation_report_*.csv`

The expert system scores each completed song on:

- Narrative clarity: lyric structure, concrete imagery, verse/chorus completeness.
- Hook strength: chorus, title recall, useful repetition, short memorable lines.
- Mainstream fit: style tags that match broad short-video music aesthetics.
- Production readiness: WAV duration, peak, rough RMS, clipping, silence.
- Platform packaging: whether it is a complete vocal song and suitable short-video length.

Scores are used for ranking and review priority. Nothing is auto-published.

## 2. Render A Vertical Lyric Video

Use a selected job id from the report:

```powershell
python tools/social_media/render_video.py --job-id fa376fbde736
```

Output:

- `outputs/social_media/<job-id>.mp4`
- `outputs/social_media/<job-id>.ass`

The video is `1080x1920`, uses the generated song audio, and overlays title plus scrolling lyrics. It depends on `ffmpeg` being available in `PATH`.

Style templates:

```powershell
python tools/social_media/render_video.py --job-id fa376fbde736 --template sweet
python tools/social_media/render_video.py --job-id fa376fbde736 --template guofeng
python tools/social_media/render_video.py --job-id fa376fbde736 --template battle
```

Use your own licensed/public-domain clips as an auto-cut background:

```powershell
python tools/social_media/render_video.py --job-id fa376fbde736 --template battle --clips-dir assets/social_clips/battle
```

The clip folder can contain `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, or `.m4v`. The script randomly cuts 2-5 second segments, crops them to vertical video, applies the selected look, and overlays lyrics.

For anime-style edits, keep the source folder limited to material you have rights to use: your own generated clips, licensed stock footage, public-domain animation, or clips you have explicit permission to publish. Do not treat this script as a copyright bypass tool.

Useful future integrations:

- PySceneDetect: best fit for splitting long animation/source videos into shot-level clips before they enter `--clips-dir`.
- auto-editor: useful for speech/talking-head content, less central for music videos because it is mainly loudness/silence driven.
- MoviePy: useful if the workflow needs Python-native compositing, image layers, stickers, or more custom effects than ffmpeg filters.

You can also render any WAV directly:

```powershell
python tools/social_media/render_video.py --audio outputs/song.wav --title "Song Title" --lyrics-file lyrics.txt
```

## 3. Package Music Queues

```powershell
python tools/social_media/package_candidates.py --min-score 85 --clean
```

Output:

- `outputs/review_queue/manifest.json`
- `outputs/review_queue/*.wav`
- `outputs/needs_work_queue/manifest.json`
- `outputs/needs_work_queue/*.wav`
- `outputs/instrumental_queue/manifest.json`
- `outputs/instrumental_queue/*.wav`

Files are renamed as:

```text
001_72_review_mixed_随想_fa376fbde736.wav
002_58_needs_work_battle_ExampleSong_abcd1234.wav
```

This keeps the original job id while making the folders usable for human review. Nothing is auto-published.

Queue rules:

- `outputs/review_queue`: score is at or above `--min-score`; these are worth listening to.
- `outputs/needs_work_queue`: score is below review threshold but still worth tracking, or the audio has production issues such as missing files, decode failure, clipping, loudness/silence problems, high persistent noise, or short/invalid duration.
- `outputs/instrumental_queue`: pure instrumental tracks are separated because the main account direction is vocal music.

For the current top-song direction, use `85` as the default review threshold. Lower scores can still enter `needs_work_queue`, but they should not be treated as publish candidates.

The category folders are intentionally broad:

- `sweet_female`: sweet female vocal, healing, dreamy, piano-forward material.
- `guofeng`: Chinese-style, guzheng/erhu/pipa, classical-pop material.
- `battle`: rock, epic, cinematic, drum-forward, combat-like rhythm.
- `mixed`: everything else that still reaches the base score.

## 4. Top Song Workflow

This is the default operating command for serious batches:

```powershell
python tools/social_media/top_song_workflow.py
```

It clears old queue files, packages high-score songs into a temporary review queue, runs vocal-stem optimization, exports the final files into `outputs/top_songs`, then removes the temporary review/instrumental queues.

The human-facing folder is:

- `outputs/top_songs/*.wav`: final repaired high-score songs.
- `outputs/top_songs/*.txt`: same-name lyrics for checking.

Machine-side folders:

- `outputs/library_audio`: original generated WAVs used by the app database.
- `outputs/needs_work_queue`: lower-quality songs retained for debugging or regeneration decisions.
- `outputs/vocal_repair/stem_pipeline`: cached separated stems and remixes so the same song does not need slow Demucs separation again.

Use this after a generation batch when the goal is to find top candidates rather than keep every acceptable song.

## 5. Render Shorts

Bilibili can use full versions, but Douyin/Kuaishou/Rednote-style posts should usually be short hooks. The short renderer detects the loudest sustained section, cuts a 5-60 second clip, adds short fade in/out, and reuses the vertical video templates.

Auto-pick a 15 second hook:

```powershell
python tools/social_media/short_video.py --job-id fa376fbde736 --duration 15 --template battle
```

Manual start time:

```powershell
python tools/social_media/short_video.py --job-id fa376fbde736 --start 58 --duration 20 --template sweet
```

With legal clip backgrounds:

```powershell
python tools/social_media/short_video.py --job-id fa376fbde736 --duration 12 --template battle --clips-dir assets/social_clips/battle
```

Outputs:

- `outputs/social_shorts/<job-id>_<duration>s_<template>.mp4`
- `outputs/social_shorts/<job-id>_<duration>s_<template>.wav`
- `outputs/social_shorts/<job-id>_<duration>s_<template>.json`

Recommended lengths:

- 6-9 seconds: hook test, meme-like rhythm point, fast A/B testing.
- 12-18 seconds: default short-video music preview.
- 25-35 seconds: stronger chorus/drop preview when the song needs build-up.
- Full song: Bilibili, YouTube, archive, or fan-listening version.

## Recommended Operating Loop

1. Generate 10-30 variants in the existing studio.
2. Run `python tools/social_media/top_song_workflow.py`.
3. Listen only to `outputs/top_songs` unless debugging is needed.
4. Manually keep only songs with strong melody, clean human-like vocal, memorable hook, and replay value.
5. Render full videos for Bilibili and archive use.
6. Render 8-30 second shorts for hook testing.
7. Upload the MP4s, then record actual account metrics back into future scoring rules.

The scorer is deliberately conservative. It is meant to filter obvious weak candidates and rank likely mainstream material, not replace final human listening.
