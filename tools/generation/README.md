# Next-Generation Song Backend

> **⚠️ 部分过时（2026-10-16）**：本文件写的默认后端是 **HeartMuLa 3B**；
> 2026-10-13 起默认后端是 **YuE2-3B**（`run_epic_yue2.py` / `yue2_batch_battle.py` / `yue2_plan_edit.py` /
> `plan_shape.py`）。其中 `clean_pipeline.py`、`loudness.py`、`audio_qa.py`、`battle_batch.py`
> 仍是**在用的库**（被 YuE2 脚本 import），其 API 描述有效。
> 计划层测量工具见 `plan_metrics.py` / `word_rate.py`（2026-10-16 新增），
> 规范见 `docs/arrangement_spec.md`。

This folder contains the redesigned source-clean generation pipeline. The old
"generate -> post-process repair" pipeline (frame repair / EQ / humanize /
de-esser) is ABANDONED — it has a hard ceiling and every repair pass traded one
artifact for another. The new pipeline generates from the source and touches
nothing.

## 2026-10-26 Chinese lyric + melody chain

The Chinese week adds a lyric gate and a symbolic melody gate in front of YuE2:

```text
zh_lyrics_gen.py  ->  zh_lyric_qa.py  ->  yue2_batch_battle.py  ->  melody_qa.py
(DeepSeek brief)      (lyric gate)         (render + plan shape)     (plan gate)
```

- `zh_lyrics_gen.py` writes a brief from a direction card (scene / addressee /
  secret / image bank / rhyme / BPM) and retries locally until the lyric gate
  passes.  It rejects English lines, onomatopoeia/humming, missing hooks,
  non-open-vowel hooks, cliche/AI-copywriter language and wrong section budgets.
- `zh_lyric_qa.py` measures section quotas, line lengths, per-section rhyme,
  3rd-tone runs, line-final tones, open-vowel endings, hook length/recurrence,
  concrete objects, abstract words and repetition.
- `melody_qa.py` reads the YuE2 ABC plan directly and measures chorus/verse
  density, register lift, final-chorus lift, stepwise/leap/repeated-note shares,
  range, climax position and chorus hook similarity.
- `yue2_batch_battle.py --melody-target zh-pop` folds the melody score into the
  candidate ranking and saves `melody_qa.json` per take.
- `zh_pop_review.py` renders the Chinese listening sheet.
- Standards and research: `docs/lyric_melody_zh_2026.md`;
  Claude Skill: `.claude/skills/mandarin-pop-lyric/SKILL.md`.

## New Pipeline (Phase 7): source-clean generation with vocal-only review cleanup

```text
lyrics + tags -> HeartMuLa (official recommended weights) -> native 48 kHz WAV
             -> linear loudness normalization -> human-listening gate
             -> optional vocal-only noise/artifact review variant
```

Principles:

1. **Source quality first.** Uses the official recommended HeartMuLa
   happy-new-year weights + HeartCodec-oss-20260123 (verified present locally;
   all shard sizes match HuggingFace). The source-clean WAV remains the
   immutable reference.
2. **Multi-candidate diversity.** The same lyrics/tags are generated 3-4 times
   with different seed / temperature / topk (cfg stays at 1.0 — see below).
3. **Review cleanup is vocal-only.** The default review variant separates the
   vocal, applies transparent noise cleanup and narrow relative 8-16 kHz
   artifact reduction, then remixes the untouched instrumental. It is never
   allowed to replace the source-clean reference automatically.
4. **Listening-first acceptance.** The user picks the take. Lightweight
   objective checks only catch broken files (duration / peak / clipping /
   near-silence). No spectral proxy (zcr etc.) can veto a take.

## Run

```powershell
.\vendor\ACE-Step-1.5\.venv\Scripts\python.exe tools\generation\clean_pipeline.py `
    --lyrics outputs\candidates\song_lyrics.txt --tags outputs\candidates\song_tags.txt `
    --title song_name --candidates 4 --seconds 30
```

Outputs go to `outputs/candidates/<title>_<stamp>/` with a `manifest.json`
recording each candidate's seed/temperature/topk and objective-check result.

## Helper tools

- `audio_qa.py` — objective QA table for a set of WAVs (clicks, dropouts, harsh
  frames). Diagnostic only; it never decides a take.
- `ab_cfg.py` — same-seed A/B across `cfg_scale` values, loudness-matched, for
  human listening. Writes a README into the output folder.
- `bench_one.py` — one-config benchmark reporting timing and peak VRAM. Useful
  for re-tuning on different hardware.

```powershell
# A/B the guidance setting on one song
.\vendor\ACE-Step-1.5\.venv\Scripts\python.exe tools\generation\ab_cfg.py `
    --lyrics outputs\batch_20260820\inputs\01_starlight_on_my_shoulder_lyrics.txt `
    --tags   outputs\batch_20260820\inputs\01_starlight_on_my_shoulder_tags.txt `
    --title starlight --seconds 30 --cfgs 1.0 1.5 2.0 3.0
# then read the numbers
.\vendor\ACE-Step-1.5\.venv\Scripts\python.exe tools\generation\audio_qa.py --dir outputs\ab_cfg
```

## GPU tuning

### RTX 3060 12 GB desktop (current, measured 2026-10-09)

- **Right-size the KV cache.** The backbone preallocates its cache for
  `max_seq_len`, which is hardcoded to 8192 regardless of song length. A 30 s
  song needs only prompt (~309) + 375 frames. `clean_pipeline` now sets
  `max_seq_len` to prompt+frames rounded up to 512. This alone drops peak VRAM
  from 10.8 GB to 8.5 GB at cfg=1.0 and frees the ~2.4 GB that guidance's
  batch-2 needs.
- **cfg_scale=2.0 (guidance ON) is the default.** `cfg_scale=1.0` disables
  classifier-free guidance entirely (single batch) — that was FORCED on the old
  8 GB card, and every song delivered before 2026-10-09 was generated that way.
  Measured on the same seed with `tools/generation/audio_qa.py`:

  | cfg | clicks | clicks (both ch) | dropouts |
  |-----|-------:|-----------------:|---------:|
  | 1.0 (old fallback) | 854 | 229 | 6 |
  | 1.5 | 1130 | 359 | 2 |
  | **2.0 (default)** | **469** | **49** | **1** |
  | 3.0 | 4927 | 720 | 0 |

  cfg=2.0 halves the clicks and nearly eliminates dropouts; cfg=3.0 over-drives
  the model (10x the clicks). Listen before trusting any of these numbers — the
  project's rule is that no spectral proxy vetoes a take.
- **codec on GPU.** `--codec-device cuda` (default) cuts the decode from 5.2 min
  to 1.0 min. Peak decode VRAM is ~10.7 GB and is roughly constant with length
  (checked at 30 s and 60 s), so the 12 GB card has room. If a decode ever OOMs,
  the pipeline retries automatically with the codec on CPU.
- Token generation runs ~10 it/s at every cfg once the cache is right-sized —
  guidance is effectively free.
- Measured end-to-end: 30 s ≈ 1.6 min, 60 s ≈ 2.5 min (warm page cache).

### 8 GB laptop (legacy, measured 2026-08-15)

- cfg had to stay 1.0 (batch 2 overflowed to page file: ~56 s/token), codec had
  to run on CPU, and long songs used a separate token-gen/decode split because
  the combined in-process pipeline crashed. These constraints no longer apply on
  the 12 GB desktop.

## Backends

- **HeartMuLa 3B** (preferred): official recommended weights verified local;
  native 48 kHz output; separate model/codec loading; lazy_load keeps 8 GB
  feasible.
- **YuE** 7B Stage 1 + 1B Stage 2: weights local; `--low_vram` works but 7B
  Stage 1 is too slow on the 8 GB card for batch production. Keep as a backup
  comparison only.

## Notes

- HeartMuLa writes native 48 kHz audio (torchaudio.save hardcodes 48000). Pass
  a `.wav` output path directly — do not convert to 44.1 kHz afterwards;
  resampling is lossy.
- Old pipeline tools (`frame_repair.py`, `humanize_vocal.py`,
  `quality_rebuild_agent.py`, etc.) are kept in `tools/vocal_repair/` for
  reference but are no longer part of the production path.

## Vocal Noise and Artifact A/B

`surgical_vocal_variant.py` is the default review tool for residual vocal noise,
nasal hiss, and metallic high-frequency artifacts. It separates only the vocal,
runs the transparent profile from `noise_clean_agent.py`, applies a relative
8-16 kHz reduction to abnormal vocal frames, then remixes the untouched
instrumental. Use `--vocal-noise-profile none` only for a diagnostic A/B. The
source-clean WAV and processed WAV are both kept; the processed version is
never silently promoted.
