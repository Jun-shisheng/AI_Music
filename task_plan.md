# AI Music Quality Rebuild Plan

> **⚠️ 归档（2026-10-16，勿作为当前计划）**：这是 2026-08-15 的 "HeartMuLa + 零后处理" 计划。
> 生成后端已在 2026-10-13 换成 **YuE2-3B**，本文件里 "HeartMuLa 是本地最终生成后端"
> 等结论已过时。当前状态见 `progress.md`，编曲/节奏规范见 `docs/arrangement_spec.md`。
> 保留原因：Phase 记录与"错误→解决"表仍是有效的工程史（尤其旧修复链为何被放弃）。

## Goal

Redesign the pipeline from the source to minimize AI noise, while retaining an immutable
it after generation. The old repair chain (frame repair / EQ / humanize) proved
to have a hard ceiling — every pass traded one artifact for another. The new
pipeline is SOURCE-CLEAN: multi-candidate generation with official recommended
weights, ZERO post-processing repair, listening-first acceptance.

## Phases

- [completed] Phase 1-5: legacy repair-chain workflow (superseded).
- [completed] Phase 7 (REDESIGN): source-clean generation plus vocal-only review cleanup. `clean_pipeline.py` retains the source-clean reference; `surgical_vocal_variant.py` applies transparent vocal cleanup and narrow artifact reduction without processing the instrumental. Acceptance is listening-first.
  - 8GB GPU tuning: cfg_scale=1.0 (single batch) + codec on CPU -> ~3.4 it/s token generation (near real-time); cfg>1.0 triggers memory swap (56s/token). 30s candidate total ~6 min.
  - First batches delivered: 3 English (silver_rain) + 3 Chinese (月光小巷) 30s candidates, all passing objective checks, zero post-processing.
  - 60s full-length works: the in-process pipeline crashes on 8GB (mula+codec memory conflict), but the SPLIT path (token gen -> save frames -> codec decode) is stable. `clean_pipeline --full-song` implements it; `heartmula_runner` auto-enables it for >=60s. Delivered 月光小巷 60s + silver_rain 60s full versions.
  - Service integrated: `GENERATION_BACKEND=heartmula` (default) routes jobs through `heartmula_runner.py` -> clean_pipeline; the legacy repair chain is never applied on this path.
  - Tests: 52 tool tests (incl. clean_pipeline 5) + 25 API tests (incl. heartmula_runner 4) all passing.
  - REMAINING: user listening selection of candidates; production batch from the chosen take's parameters.
- [completed] Phase 8 (20-song bilingual batch): generated ten Chinese and ten English 30-second HeartMuLa review candidates with one resident model, then ran vocal-only transparent cleanup, narrow 8-16 kHz artifact reduction, objective checks, and listening review packaging. Finalized 20/20 WAV + MP3 + lyrics TXT deliverable sets; 19 used vocal-clean variants and 1 used the source fallback.

## Acceptance Criteria (new design)

- The source-clean reference is always retained.
- Any cleanup variant touches only the separated vocal; the instrumental is remixed unchanged.
- Vocal review cleanup is transparent by default and never auto-promoted without listening.
- 3-4 candidates per lyrics/tags are generated with parameter diversity.
- Broken files (duration/peak/clipping/near-silence) are rejected objectively.
- The human picks the final take by listening; no spectral proxy can veto it.
- Outputs are native 48 kHz WAVs; every review WAV has matching lyrics TXT.
- Temporary raw files are deleted after normalization.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| None in this redesign session yet | 0 | Pending |
| Long-track transparent cleanup exceeded 15 minutes and left an orphan FFmpeg process | 1 | Added a 180-second timeout, automatic source-vocal fallback, and switched this workflow to the fast transparent filter chain |
| Production chain final output still had 9 harsh events after frame repair had reduced them to 1 | 1 | Root-caused: de_buzz EQ's 7 kHz air shelf re-inflates 4-10 kHz when applied after frame repair. Reordered to noise floor → EQ → high-note tame → frame repair LAST → 0 harsh events, gate 97 |
| User still heard noise after the reorder; single frame-repair pass left 4 of 18 harsh events (zcr 0.65→0.56-0.68, still above 0.55) | 1 | Two repair rounds with RE-DETECTION between rounds (re-applying the mask without re-detection only reaches 3). Dry stem 18→0; final-mix re-separated proxy 11→1, harsh_ratio 0.041→0.027 |
| User then reported unclear diction after the double-pass fix | 1 | Root-caused: the zcr harsh detector cannot separate sibilants from machine noise (human refs trigger 14-149 events). Standard repair ate 5-8k by 1.6-1.8 dB. New isolation_only production mode repairs only isolated single-frame bursts → 5-8k loss halved to -1.0 dB while removing true burst noise; aggressive double-pass kept as variant repair_std2 |
| User asked to abandon the repair pipeline entirely and redesign from source; delete all non-conforming outputs | 1 | Started Phase 7: verified local weights are the latest official versions, deleted outputs/, wrote clean_pipeline.py (multi-candidate, zero post, listening-first) |
| Isolated 20-song generation repeatedly exhausted CUDA/page-file resources while ComfyUI was active | 2 | Waited for ComfyUI/resource release, then switched to one resident HeartMuLa process so the 15.7 GB checkpoint loads once. Generation is stable without OOM. |
| Batch finalization had to avoid promoting cleanup when objective metrics regressed | 1 | Added automatic source fallback; all 20 outputs were packaged, with one title retained from the source-clean reference. |

## Decisions

- RVC remains disabled.
- SoulX full-song SVS is not a default path because listening exposed heavy electronic timbre and machine analysis found phrase dropouts.
- The legacy repair chain (frame repair / EQ / humanize) is ABANDONED as the default path — it has a hard ceiling and every pass trades one artifact for another.
- The new pipeline is source-clean: official recommended HeartMuLa happy-new-year + HeartCodec-oss-20260123 (verified local), multi-candidate diversity, zero post-processing, listening-first acceptance.
- ACE-Step Turbo is draft-only for the target quality level.
- 2026-08-20: seed propagation is now real, and vocal-only surgical HF reduction is available as an explicit A/B path. It never silently replaces the source-clean output.
- 2026-08-20: vocal-only review cleanup is now the default surgical path: transparent FFmpeg cleanup followed by relative HF artifact reduction. The instrumental remains untouched and `--vocal-noise-profile none` remains available for diagnostic A/B.
- YuE 7B Stage 1 is too slow on the local 8GB GPU for batch production; HeartMuLa 3B is the local final-generation backend.
- Local HeartMuLa weights ARE the official recommended versions — no re-download needed (all shard sizes match HF).
