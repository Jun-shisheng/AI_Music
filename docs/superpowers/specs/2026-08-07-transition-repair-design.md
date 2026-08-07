# Transition Repair (转音修复) Design

Date: 2026-08-07
Status: approved by user

## Problem

User's last listening feedback (2026-08-06): at 27s/40s/50s/end of the HeartMuLa 60s
candidate, pitch transitions ("转音") are steppy and broken. F0 analysis of the vocal
stem confirmed three concrete defect classes at transition points:

| # | Defect | Evidence (HeartMuLa 60s vocal stem) |
|---|---|---|
| 1 | True-silence gap between notes (dropout) | 38.6-40.1s: ~1.3s of -80dB silence while the accompaniment plays continuously; a faint note (-70dB) ends at 38.62s, the next phrase starts rising at 40.15s |
| 2 | Pitch step without glide | 49.5s: previous note ~330Hz ends, next note starts directly at 482Hz; 55-59s: multiple +4..+7 st instantaneous jumps |
| 3 | Screech / unpitched-noise segment between notes | 49.5-49.8s: ~300ms of strong energy (-17dB), peak ~700Hz, high aperiodicity — pyworld harvest cannot lock F0; the noise floor reducer and frame repair do not fix it because the silence expander even *extends* the 39s gap (faint -70dB note is treated as silence) |

The existing four-layer chain (frame_repair -> noise_floor_reducer -> high_note_tamer
-> de_buzz) cannot fix these: frame repair leaves silent gaps untouched by design,
noise floor reduction widens the 39s gap, and none of them can produce a pitch glide.

## Approach (selected: A)

Surgical WORLD resynthesis of transition windows only. Everything outside a detected
transition window is bit-exact original audio. Alternatives rejected:

- B (extend existing chain only): cannot fix pitch steps; rejected.
- C (full-stem WORLD resynthesis with globally smoothed F0): whole-stem texture loss,
  kills the de_buzz air shelf; rejected (previously flagged high risk).

## Module: `tools/vocal_repair/transition_repair.py`

Three independent units with clear interfaces, tested separately.

### 1. Event detector

Input: vocal stem (any sr, mono-converted), pyworld F0 at 5ms frame period.

- **step** (硬跳变): two stable notes (>=150ms, pitch within +/-1 st of running
  median) separated by <=300ms with |delta_st| >= 2.0 where the pitch change is
  concentrated in <=5 consecutive frames (no glide). Event window = gap between
  notes, with a minimum window of 100ms centered on the transition point when
  the notes are adjacent (zero-length gap).
- **silence_gap** (静音断裂): notes with true silence between them. Energy is
  measured on the **middle** of the inter-note window (adaptive margins of
  min(100ms, 30% of the window) per side, clamped windows): the middle RMS
  median < -60dB classifies silence_gap (median over max, because note-onset
  audio can contaminate up to ~370ms into the F0 gap window on real stems).
  Gap <= `--max-gap-ms` (default 300ms).
- **screech** (尖啸): >=60ms segment between two voiced notes with energy in
  the middle of the window (25ms RMS max >= -45dB) but F0 == 0 or erratic
  (harvest-locked noise: std of st_of over voiced frames > 2.5 st OR mean
  |frame-to-frame st diff| > 2.0 st). Event window = segment + short margins.

Longer gaps (> max-gap-ms, e.g. the 1.3s at 39s) surface as `silence_gap`
events but are **not** bridged automatically; they are flagged in the report
JSON as `long_gap_skipped` so the user can opt in via a larger `--max-gap-ms`
for A/B listening. Screech windows (energy-bridged) are always repaired
regardless of length.

Output: `[{type, t0, t1, f_a, f_b, flags}]` (times in seconds, f in Hz).

### 2. Glide F0 builder

Input: event + note pitches (f_a at t_a, f_b at t_b).

- Log-frequency linear interpolation (exponential glide) from f_a to f_b over a
  ~150ms window (configurable 100-250ms), clipped to the event window.
- Natural shape: hold f_a for the first ~30%, glide, +0.3-0.5 st overshoot at
  ~80% of the window, settle to f_b at the end.
- Edge convergence: the last/first 20-30ms of the glide converge to the note F0
  so the time-domain crossfade has no pitch step.

Output: per-frame F0 array over the event window.

### 3. WORLD synthesis and blending

- WORLD analysis of the full vocal stem: harvest F0, cheapTrick spectral envelope,
  D4C aperiodicity (5ms frame period).
- Per event window:
  - Replace F0 frames with the glide F0 (voiced throughout the window).
  - silence_gap: envelope interpolated (log-domain) from note A's last frames to
    note B's first frames, volume ramping in dB so the bridge "slides into" the
    next phrase.
  - screech: aperiodicity blended toward the surrounding notes' values (de-scream);
    envelope kept from the window's own analysis (blend 0.5 original + 0.5
    interpolated as a tuning knob).
  - step: envelope and aperiodicity from the window's own analysis (the gap has
    audio; only pitch trajectory is replaced).
  - pyworld.synthesize for the window only.
- Overlap-add: synthesized window placed back with 30ms crossfade at each edge.
- Construction guarantee: only event-window samples are overwritten; everything
  else is bit-exact original.

## Pipeline integration

Position: `demucs -> frame_repair -> noise_floor -> [transition_repair] -> de_buzz -> remix`

- Runs after noise_floor so WORLD analyzes clean audio and the 49.5s screech has
  already been tamed by frame repair.
- The synthesized glide passes through the same de_buzz EQ, keeping timbre
  consistent with the rest of the vocal.

Integration points (same pattern as the humanize chain):

1. `quality_rebuild_agent.py`: transition repair as a candidate step in the
   humanize chain, listening-first preference (chosen when it does not regress
   gate/structural/dropout; +4 artifact tolerance as with humanize).
2. `queue.py` FastAPI hook: `VOCAL_TRANSITION_REPAIR_ENABLED=true` (default on),
   falls back to raw output on failure.
3. CLI: `transition_repair.py <vocals.wav> <out.wav> [--max-gap-ms] [--report report.json]`
   plus event report output.

## Testing

- Unit tests (synthetic signals, sine + vibrato) with engineered step / silence_gap
  / screech transitions assert:
  - events detected with correct type and window,
  - output F0 follows the glide (re-extract with pyworld),
  - samples outside windows are bit-identical to input,
  - crossfade edges have bounded sample-to-sample delta (no click).
- Gate regression: full 60s HeartMuLa run through the whole chain; artifact / gate
  / noise metrics must not regress (with the +4 artifact tolerance).
- A/B: user listens to the 4 transition points (original vs repaired clips) plus
  the full song before the module is accepted into the default pipeline.

## Error handling / fallbacks

- WORLD import failure: skip transition repair, warn (same pattern as humanize
  falling back to raw).
- Per-event synthesis failure: try/except, keep the original window.
- No events detected: bit-exact pass-through.

## Out of scope

- RVC, SoulX, backend regeneration, whole-song structural changes.
- Bridging musically long phrase breaks beyond `--max-gap-ms` without user opt-in.
