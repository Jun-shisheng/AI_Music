# Findings

> **⚠️ 归档（2026-10-16，勿当作当前管线）**：这是 2026-08-05 ~ 08-08 的缺陷解剖记录，
> 对象是当时的 **ACE-Step / HeartMuLa** 后端，后处理链（frame repair / humanize / de_buzz）
> 已于 2026-08-15 被整体放弃。当前后端是 **YuE2-3B**，当前状态见 `progress.md`。
> **保留原因（重要）**：这是"AI 人声杂音"的根因知识库——harsh 帧的 zcr/频段特征、
> 噪声底不是问题所在、齿音与机器噪声为何难分辨、EQ 与降噪的先后顺序为何关键。
> 2026-10-16 已实测：这些检测器在 YuE2 输出上**全部失效**（harsh 恒为 0），
> 见 `docs/arrangement_spec.md` §2。

## Existing Evidence

- Whole-mix filtering can make accompaniment sharper and thinner.
- RVC changed timbre but did not remove source artifacts.
- SoulX-SVC reduced proxy artifact scores but remained audibly synthetic.
- SoulX-SVS produced acceptable 30-second diction, but full-song output had heavy electronic character and abrupt vocal level drops.
- Existing project already has Demucs separation, noise cleanup, vocal artifact analysis, scoring, and shallow review-folder conventions.

## Working Hypothesis

The best near-term quality gain comes from selecting clean material at segment level and applying conservative cleanup only where the defect is actually noise. Structural synthesis defects must fail the gate or be replaced from another candidate.

## Audit Results

- `outputs/top_songs` contains repeated takes of two English titles, so whole-song candidate selection is immediately usable.
- Existing artifact, noise, timestamp-event, and vocal-quality analyzers cover complementary failure modes.
- Instrumental stems must remain passthrough. Vocal cleanup is discarded automatically if post-clean metrics regress.
- Segment replacement is unsafe unless arrangements are demonstrably aligned. The first production version selects complete takes and does not splice unrelated generations.

## Execution Results

- `Postcards From Tomorrow`: rejected at artifact 37 with 12 structural events; cleanup increased dropout duration.
- `月下长安`: rejected at artifact 27 with 8 structural events; cleanup increased dropout from 0.30s to 0.55s.
- `Postcards After Midnight`: closest legacy take, but rejected at artifact 25 with 5 structural events.
- Three `Someone I Almost Knew` takes were all rejected; the best measured artifact 37 with 5 structural events.
- The configured ACE-Step LoRA points at the legacy high-score set. It must not be treated as a clean-vocal adapter until its training data passes the new vocal gate.

## New Batch Results

- `Moonlit Silk`: artifact 9, vocal gate 100, structural events 1, dropout 0.05s.
- `Run Into The Dawn`: artifact 14, vocal gate 100, structural events 2, dropout 0.05s.
- `Sweet Little Fire`: artifact 19, vocal gate 98, structural events 2, dropout 0.15s.
- All three passed the strict final gate. In each case the cleanup candidate was rejected by before/after regression checks, so no aggressive noise filter was applied.
- User listening identified residual tremolo and electronic noise despite the proxy pass. The gate now also requires `noise_events <= 8`; this is a deliberate human-listening calibration and will reject the three prior review files on the next rebuild.
- ACE-Step reference-audio test: best candidate reached artifact 18 and gate 97, but still had 9 noise events. Human vocal reference improves the result slightly but does not remove the model's synthetic vocal ceiling.
- The production target is now a model A/B benchmark, with YuE quantized as the first replacement candidate and SongGeneration 2 as the higher-memory comparison. ACE-Step Turbo remains a fast draft backend only.
- YuE setup is viable locally: X-Codec is present, CUDA/FlashAttention/Transformers import successfully in the ACE CUDA environment, and the 1B Stage 2 checkpoint is complete. The 7B Stage 1 checkpoint still needs a low-VRAM-compatible download/inference test.
- YuE Stage 1 is now fully downloaded and loads with `device_map=auto`, using roughly the available 8 GB GPU budget plus CPU offload. The 100-token Stage 1 smoke output is valid, but the low-VRAM generation rate is not yet suitable for 3.5-minute songs.
- A standalone short Stage 2 decoder now handles cached clips below the official 6-second block size. This proves stem reconstruction and mixing, but the 1-second smoke audio is a pipeline artifact, not a quality candidate.
- The first HeartMuLa 30-second candidate is not yet good enough for the human-like target. Its machine metrics are better than some old ACE samples, but not sufficient to claim natural vocals; the final standard remains listening-first.

## 2026-08-05 Defect Autopsy

- Localized the audible "noise + breaks" to three concrete defect classes found on BOTH backends (ACE-Step and HeartMuLa), always on the separated vocal stem:
  - harsh_machine_noise: 50 ms frames with zero-crossing ratio 0.38-0.79 and an abnormal 4-10 kHz energy burst (+25-30 dB over normal singing). Normal frames sit at 0.10-0.20 zcr; defect frames dominate the "electronic noise" listening impression.
  - abrupt_level_drop: single-frame 10-13x level collapse (vocal "break").
  - possible_dropout: 50-400 ms low-energy gaps between active frames (broken word/syllable).
- Whole-track noise floor is NOT the issue: HF noise floor is -93 to -112 dB on both backends. Any broadband filter approach (afftdn etc.) is the wrong tool and was confirmed again to regress the gate (cleanup variant scored 71 vs 88 source on HeartMuLa).
- HeartMuLa native output is 48 kHz (heartlib torchaudio.save hardcodes 48000; HeartCodec config 48000). The 44.1 kHz candidate WAV was a manual conversion artifact; resampling was ruled out as a noise source (48k mp3 vocal stem has the same defect timestamps).
- Full-length songs have ~1 harsh frame per second (ACE-Step: 212 frames/210s, mostly fringe 0.38-0.41), so any top-N-truncated event list under-covers full songs. Event detection must run uncapped for repair coverage.
- New `tools/vocal_repair/frame_repair.py` repairs defect regions on the vocal stem only: STFT-domain 4-10 kHz band attenuation (0.22 factor, -13 dB) inside detected frames with crossfaded edges, plus gain recovery for level drops/dropouts with residual signal. Silent gaps stay rejected, not patched.
- Calibrated noise_events to count only severity >= 0.55 harsh frames (audible events), not threshold-fringe frames. Full-track counting with top-12 truncation was rejecting every local-model song regardless of audibility.
- Measured results after integration into quality_rebuild_agent:
  - Moonlit Silk: harsh 0.114 -> 0.039, noise_events 129 -> 8, artifact 17 -> 8, gate 96 -> 100. PASSED.
  - Run Into The Dawn: harsh 0.069 -> 0.038, noise_events 71 -> 5, gate 100. PASSED.
  - Sweet Little Fire: harsh 0.110 -> 0.050 but noise_events 16 > 8. REJECTED (candidate selection should pick the best take).
  - HeartMuLa 30s smoke: noise_events 10 -> 1, dropout 0.4 -> 0.25, gate 88 -> 92. PASSED (first HeartMuLa candidate to pass).
- Frame repair preserves non-defect regions exactly (tested); artifact score does not regress. The remaining ceiling is the models' synthetic vocal texture itself; frame repair lowers the noise floor of defects but cannot turn the texture fully human.

## 2026-08-05 Second Round: Timbre Autopsy

- User follow-up: "all vocals still have an electro feel; worse tearing when pitch goes up (~47s chorus)". Root-caused the TIMBRE layer, separate from the earlier noise/break layer.
- Ruled out: F0 jitter (pyworld: 0.6-1.1% at all windows vs human 1-3% — the earlier 18% was a crude autocorrelation estimator artifact; pitch stabilization was built then discarded as unnecessary), clipping (peak 0.55 at chorus, 0% near-clip), loudnorm pumping (remix envelope modulation <= original), stereo phase (median corr 0.81, p10 -0.05), and the ACE-Step LoRA (log evidence: LoRA load returned 400 every attempt -> the service has always generated with the BASE model; the LoRA is not a timbre factor today).
- LTAS vs a real human vocal reference (human_refs/vocals/en): the AI vocal is NOT hot in the mid band — it is 2-7 dB QUIETER than human in 3-12 kHz (missing "air"/breath), slightly boxy at 1-2k, and MORE periodic (HNR higher, spectral flatness 0.06 vs 0.11) — the "electro" character is the model's too-clean, airless texture, not excess noise.
- User A/B listening over 5 full-song vocal treatments (baseline / air EQ / air+soft-saturation / air+deesser / air+reverb) selected **de_buzz**: natural formant EQ (cuts 280/430/950/3600 Hz) + 7 kHz air shelf + deesser + limiter.
- The air shelf inflates the harsh-frame noise and artifact proxies by design (+2 artifact, +20-40 noise frames) without audible regression — listening-first selection now uses a humanize-specific regression check (artifact +4 tolerance, gate/structural/dropout strict) and a relaxed noise ceiling (120) when the humanize chain is selected.
- Production integration: `tools/vocal_repair/humanize_vocal.py` (single-variant mode), `quality_rebuild_agent.py` (humanize candidate + listening-first preference), and the FastAPI service (`queue.py` post-generation hook, `VOCAL_HUMANIZE_ENABLED=true` default) so every generated song gets demucs -> frame repair -> de_buzz EQ -> remix automatically.
- Rebuild results with humanize preference: Moonlit Silk (humanize), Run Into The Dawn (frame — humanize regressed artifact), Sweet Little Fire (humanize) — all three now pass (Sweet Little Fire was previously rejected). HeartMuLa 30s passes with humanize (artifact 12, gate 89, noise 1).
- The model's vocal texture ceiling remains: post-processing reshapes toward the human envelope but cannot synthesize true breathiness/vocal fry. The final-humanity step is still the generation-backend benchmark (YuE / HeartMuLa) with this humanize chain applied on top.

## 2026-08-08 Production-Chain Order Root Cause

- Stage-wise harsh-event measurement on a 60s HeartMuLa vocal stem: demucs 11, after noise-floor reducer still 11, after frame repair **1**, but the production chain's final output was back to **9**.
- The de_buzz EQ chain ends with a 7 kHz air shelf (+2.5 dB). Applied AFTER frame repair, the shelf re-inflates the 4-10 kHz band that repair had suppressed — the fix was being undone by the very next stage.
- Reordered chain (noise floor -> de_buzz EQ -> high-note taming -> frame repair LAST) measures gate 97 vs 94, harsh events 0 vs 9, dropout 0.3s vs 0.4s on the same 60s sample. The high-note tamer contributes the last event (1 -> 0).
- Rule: any EQ/shelf/air stage that boosts high frequencies must run BEFORE frame repair; frame repair is the terminal vocal stage.
- Remix note: the final remix applies loudnorm, whose gain lifts quiet frames above the harsh-detector activity floor, re-revealing latent harsh frames (0 with no loudnorm vs 3 with it on the same 60s stem). Re-separating the final mix with demucs also injects separation residue that the dry stem does not contain (e.g. an 11.90s event appears only after mix re-separation). The pre-remix vocal stem is the cleanest metric: raw 12 harsh events -> fixed chain 4, gate 94 -> 97.
- Single-pass frame repair is insufficient: it lowers harsh-frame zcr from ~0.65 to 0.56-0.68, still above the audible 0.55 threshold. The repair input (after noise floor + EQ + tame) carried 18 harsh events; one pass left 4. A second round reaches 0 with negligible band change (4-10k -0.4 -> -0.5 dB, 1-4k and 0.2-1k identical). Final-mix re-separated vocal proxy: raw 11 -> single round 2 -> double round 1 harsh events, harsh_ratio 0.041 -> 0.027.
- Implementation detail: the second repair round MUST re-detect events on the current signal (frames a first round pushed to zcr 0.56-0.68 are still above the 0.38 detector floor). Re-applying the same mask to the same regions without re-detection only reaches 3 events, not 0.
- The zcr harsh detector cannot separate sibilants from machine noise: human vocal references trigger 14-149 'harsh' events per file (s/sh/ch naturally have high zcr) while AI stems trigger only 17. Standard 1-2 round repair therefore eats the 5-8k sibilant band by 1.6-1.8 dB -> unclear diction. Isolation-only repair (production default) fixes only isolated single-frame bursts (AI harsh frames 68% isolated vs 27% for human sibilant runs), halving the 5-8k loss to -1.0 dB while still removing true burst noise. Aggressive double-pass remains available as humanize variant `repair_std2`.
- The noise is NOT a broadband floor problem: quietest 0.5s segments of the final mix already sit at -75..-100 dB per band. The audible noise is localized harsh bursts inside sung phrases, plus loudnorm lifting of latent frames.
- Residual dropout ~0.3s remains near the 0.35s gate ceiling and is a backend-model property, not a post-processing artifact.
