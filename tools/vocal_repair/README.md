# Vocal Repair Workflow

> **⚠️ 已废弃（2026-08-15 起，2026-10-16 归档确认）**：整套 frame repair / humanize / RVC /
> SoulX 后处理链已被放弃——它被证明有硬天花板（每次修补都用一种瑕疵换另一种）。
> 当前原则是**源头干净、零修补**，后端为 YuE2-3B。唯一仍有参考价值的是
> `frame_repair.py` 等文件里的检测思路，以及 `prepare_human_refs.py`（用它做参考曲的人声分离）。
> 注意：`services/api/app/services/queue.py` 里还留着对 `humanize_vocal.py` 的调用，
> 那是历史服务里的陈旧引用，不是当前路径。

This subproject is for generated songs that already reached the review threshold but still need better vocal presentation.

It does not try to magically reconstruct broken syllables. It does three practical jobs:

- Detect likely vocal dropouts, abrupt level breaks, clipping, and harsh high-frequency artifact frames.
- Produce diagnostics for whether a passed song needs vocal-only repair or regeneration.
- Mark severe cases as `needs_regen` so they are regenerated instead of being polished endlessly.

## Humanize Chain (production FastAPI path)

The production post-generation hook (`queue.py` -> `humanize_vocal.py --variant de_buzz`)
runs this exact order on the demucs-separated vocal stem:

```text
demucs vocals -> noise floor reduction -> de_buzz EQ (formant + 7k air shelf + deesser)
             -> high-note taming -> frame repair (LAST) -> remix with instrumental
```

**Order matters (measured root cause):** the de_buzz chain's 7 kHz air shelf
re-introduces harsh high-frequency noise when it runs AFTER frame repair. On a
60s HeartMuLa vocal stem the old order (repair -> EQ) left 9 audible harsh
events and gate 94; the fixed order (EQ -> repair last) leaves 0 harsh events
and gate 97, with dropout 0.4s -> 0.3s. Frame repair must always be the final
vocal step so defect regions are suppressed after the EQ stage.

**Two repair rounds (measured):** a single frame-repair pass only lowers harsh
frame zcr from ~0.65 to 0.56-0.68, which is still above the audible 0.55
threshold; the second round pushes those frames under 0.55. Band energy is
essentially unchanged (4-10k: -0.4 -> -0.5 dB, 1-4k and 0.2-1k identical), so
the extra round is local and safe.

**Isolation mode is the production default (diction-preserving).** The zcr
harsh detector cannot separate sibilants from machine noise: human vocal
references trigger 14-149 'harsh' events per file because s/sh/ch naturally
have high zcr (AI stems trigger only 17). Standard repair (1-2 rounds) eats the
5-8k sibilant band by 1.6-1.8 dB, which users hear as unclear diction.
`repair_vocals(..., isolation_only=True)` repairs only ISOLATED single-frame
bursts (AI harsh frames are 68% isolated vs 27% for human sibilant runs) and
preserves multi-frame runs (consonants). Measured on the 60s HeartMuLa stem:
5-8k diction loss halves to -1.0 dB while isolated burst frames are still
removed. `humanize_stem` defaults to isolation mode; `frame_repair.py` CLI
exposes `--isolation-only` and `--rounds`.

`process_full_song.py` runs the same chain through the CLI; `quality_rebuild_agent.py`
uses `humanize_stem` so it inherits the fixed order automatically.

## Run

```powershell
python tools/vocal_repair/vocal_repair.py
```

Input:

- `outputs/review_queue/*.wav`

Output:

- `outputs/vocal_repair/reports/vocal_repair_report_*.json`
- `outputs/vocal_repair/reports/vocal_repair_report_*.csv`

The full-mix workflow is experimental because it touches the backing track and can make the accompaniment sharper. Only write it for diagnostics:

```powershell
python tools/vocal_repair/vocal_repair.py --write-fullmix --profile transparent
```

Default mode analyzes without writing repaired audio:

```powershell
python tools/vocal_repair/vocal_repair.py
```

## Interpretation

- `pass`: no hard artifact found by proxy analysis.
- `repair_recommended`: use the repaired copy for listening comparison.
- `needs_regen`: likely broken vocal phrasing or severe artifact; regenerate the song or that direction.

## Current Limits

This first version works on the mixed master because the local generator outputs a single WAV. The next level should add source separation so vocal-specific repair can run on an isolated vocal stem, then remux with the instrumental stem.

## Stem Pipeline

The preferred workflow is stem-based and inspired by the Timbre Distillation approach:

```text
review_queue WAV -> vocal separation -> noise-clean stems -> repair vocal stem -> optional RVC -> remix with instrumental
```

Run:

```powershell
.\tools\vocal_repair\.venv\Scripts\python.exe tools/vocal_repair/stem_pipeline.py
```

Outputs:

## Default Quality Rebuild Workflow

Use this path before any voice-conversion or re-singing experiment:

```text
multiple full-song takes -> cheap pre-screen -> Demucs vocal analysis
-> optional bounded vocal-only cleanup -> before/after rollback gate
-> strict structural gate -> shallow review folder
```

Run up to three takes per title:

```powershell
.\tools\vocal_repair\.venv\Scripts\python.exe tools\vocal_repair\quality_rebuild_agent.py --candidate-limit 3
```

Review outputs are written to `outputs/quality_rebuild_review` with matching lyric TXT files.
The accompaniment stem is never filtered. Cleanup is adopted only when noise proxies improve without
artifact, dropout, or structural regression. Temporary stems are deleted by default. A title with no
passing take appears only in `quality_manifest.json`; no misleading review WAV is created.

RVC, SoulX-SVC, and SoulX-SVS are experimental backup paths and are not part of this default workflow.

Outputs:

- `outputs/vocal_repair/stem_pipeline/<song>/demucs/.../vocals.wav`
- `outputs/vocal_repair/stem_pipeline/<song>/demucs/.../no_vocals.wav`
- `outputs/vocal_repair/stem_pipeline/<song>/noise_agent/vocals_noise_clean.wav`
- `outputs/vocal_repair/stem_pipeline/<song>/noise_agent/no_vocals_noise_clean.wav`
- `outputs/vocal_repair/stem_pipeline/<song>/repaired_vocals.wav`
- `outputs/vocal_repair/stem_pipeline/<song>/remix.wav`
- `outputs/vocal_repair/stem_pipeline/<song>/metadata.json`
- `outputs/top_songs/<song>.wav` when run through `tools/social_media/top_song_workflow.py`

This requires Demucs in the active Python environment:

```powershell
pip install demucs
```

On this machine the intended environment is:

```powershell
.\tools\vocal_repair\.venv\Scripts\python.exe
```

Future RVC/timbre step:

```text
vocals.wav -> vocal cleanup -> RVC/timbre conversion -> remix.wav
```

## Noise Clean Agent

RVC is timbre conversion, not denoising. If the source has metallic buzz, clicks, hiss, or separation residue, RVC can preserve or exaggerate those defects. The noise agent is a conservative cleanup pass before vocal repair or timbre conversion:

```powershell
.\tools\vocal_repair\.venv\Scripts\python.exe tools\vocal_repair\noise_clean_agent.py --vocals path\to\vocals.wav --instrumental path\to\no_vocals.wav
```

Profiles:

- `transparent`: default; removes obvious clicks, harsh hiss, and low-level non-musical residue.
- `balanced`: stronger cleanup when artifacts are audible.
- `aggressive`: diagnostic only; may dull vocals or backing texture.

### Enable RVC Manually

RVC is off by default. Enable it only for a deliberate timbre-conversion experiment:

- Python: `G:\Cover_vision\.venv\python.exe`
- CLI: `G:\Cover_vision\tools\rvc_infer.py`
- Model (preferred): `...\weights\vp_v2_48k.pth`（导出推理权重；不要用 `*_e20_s*.pth` 训练快照）
- Index: `...\logs\vp_v2_48k\added_IVF256_Flat.index`

```powershell
.\tools\vocal_repair\.venv\Scripts\python.exe tools/vocal_repair/stem_pipeline.py --force --limit 1 --rvc
python tools/social_media/top_song_workflow.py --force-optimize --rvc
```

Override root with `COVER_VISION_ROOT`, or disable auto with `RVC_DISABLE_AUTO=1`.

Manual template (optional):

```powershell
$env:RVC_INFER_CMD='G:\Cover_vision\.venv\python.exe G:\Cover_vision\tools\rvc_infer.py --model_path {model} --index_path {index} --input {input} --output {output} --f0_method rmvpe --f0_up_key 0 --index_rate 0.75 --device cuda'
```

`--rvc` means **require** success. Without it, RVC is skipped.

The practical rule is: use the noise agent and stem repair for roughness; use RVC only for deliberate timbre experiments. Broken syllables -> regenerate. Next phase: ACE-Step LoRA on curated high-score songs.

## Vocal Re-Singing Agent

When a vocal is not merely noisy but structurally fake, broken, or robotic, do not keep applying stronger filters. Use the re-singing planner:

```powershell
.\tools\vocal_repair\.venv\Scripts\python.exe tools\vocal_repair\vocal_resing_agent.py --input-dir outputs\top_songs --limit 3 --force
```

Outputs:

- `outputs/vocal_resing/<song>/demucs/.../vocals.wav`
- `outputs/vocal_resing/<song>/demucs/.../no_vocals.wav`
- `outputs/vocal_resing/<song>/vocal_segments/*.wav`
- `outputs/vocal_resing/<song>/resing_manifest.json`
- `outputs/vocal_resing/resing_report_*.json`

Decisions:

- `repair_only`: the vocal stem is acceptable enough for light repair.
- `partial_resing`: only marked phrase segments should be re-sung or converted.
- `full_resing`: the whole vocal should be treated as a melody draft and re-sung.

Optional RVC segment experiment:

```powershell
.\tools\vocal_repair\.venv\Scripts\python.exe tools\vocal_repair\vocal_resing_agent.py --input-dir outputs\top_songs --limit 1 --execute-rvc
```

This runs RVC only on segments marked `resing_required`. It is still a timbre-conversion experiment, not a guarantee of natural singing. The manifest is the contract for a future dedicated singing voice model: preserve timing and melody from the segment, regenerate a cleaner human-like vocal, then remix with `no_vocals.wav`.

## SoulX-SVS Re-Singing Workflow

The main high-quality test path is SoulX-Singer SVS re-singing. It treats the accepted AI vocal as a melody/timing draft, rebuilds the singing voice through SoulX metadata, and remixes the generated vocal with the separated instrumental.

```text
high-score song -> Demucs vocals/instrumental -> SoulX metadata preprocess -> SoulX SVS vocal -> remix -> review folder
```

Run the first 30-second review preview:

```powershell
.\tools\vocal_repair\.venv\Scripts\python.exe tools\vocal_repair\soulx_svs_workflow.py --input-dir outputs\quality25_trial --review-dir outputs\svs_review --limit 1 --preview-seconds 30 --force --fp16
```

Run a 3-song preview batch after the first preview is worth listening to:

```powershell
.\tools\vocal_repair\.venv\Scripts\python.exe tools\vocal_repair\soulx_svs_workflow.py --input-dir outputs\quality25_trial --review-dir outputs\svs_review --limit 3 --preview-seconds 30 --force --fp16
```

If the generated vocal has nasal or mechanical humming color, run the natural vocal polish variant:

```powershell
.\tools\vocal_repair\.venv\Scripts\python.exe tools\vocal_repair\soulx_svs_workflow.py --input-dir outputs\quality25_trial --review-dir outputs\svs_review --limit 1 --preview-seconds 30 --force --fp16 --no-clean-review --vocal-polish-profile natural
```

If diction needs to stay clearer, use `clear_natural` instead:

```powershell
.\tools\vocal_repair\.venv\Scripts\python.exe tools\vocal_repair\soulx_svs_workflow.py --input-dir outputs\quality25_trial --review-dir outputs\svs_review --limit 1 --preview-seconds 0 --force --fp16 --no-clean-review --vocal-polish-profile clear_natural
```

Run a full song only after the 30-second preview is approved:

```powershell
.\tools\vocal_repair\.venv\Scripts\python.exe tools\vocal_repair\soulx_svs_workflow.py --input-dir outputs\quality25_trial --review-dir outputs\svs_review --limit 1 --preview-seconds 0 --force --fp16
```

Human review output:

- `outputs/svs_review/*_soulx_svs_preview.wav`
- `outputs/svs_review/*_soulx_svs_preview_vocals.wav`
- `outputs/svs_review/*.txt`

The workflow uses:

- SoulX repository: `G:\Cover_vision\SoulX-Singer`
- SoulX Python: `D:\Miniconda3\download\envs\soulxsinger\python.exe`
- Output: `outputs\svs_review`
- Temporary work: `outputs\svs_work`, deleted by default

## Automatic SoulX-SVC Backup Workflow

RVC segment replacement is not part of the default path. SoulX-Singer SVC is a backup comparison path only:

```text
high-score song -> Demucs vocals/instrumental -> SoulX F0 extraction -> SoulX SVC vocals -> remix -> review folder
```

Run a fast 30-second review batch:

```powershell
.\tools\vocal_repair\.venv\Scripts\python.exe tools\vocal_repair\soulx_svc_workflow.py --input-dir outputs\top_songs --review-dir outputs\soulx_review --limit 3 --preview-seconds 30 --n-steps 16 --force
```

Run a full song only after the 30-second preview is approved:

```powershell
.\tools\vocal_repair\.venv\Scripts\python.exe tools\vocal_repair\soulx_svc_workflow.py --input-dir outputs\top_songs --review-dir outputs\soulx_review --limit 1 --preview-seconds 0 --n-steps 32 --force
```

The workflow uses:

- SoulX repository: `G:\Cover_vision\SoulX-Singer`
- SoulX Python: `D:\Miniconda3\download\envs\soulxsinger\python.exe`
- Output: `outputs\soulx_review`

Temporary files are removed by default. Use `--keep-work` only for debugging.
