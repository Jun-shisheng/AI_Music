# High-Quality Vocal Rebuild Plan

Date: 2026-07-20

## Objective

Build a practical path from good AI music demos to publishable songs with more natural, human-like lead vocals.

The goal is not to keep polishing broken AI vocals. The workflow should:

- Preserve high-score songs and their arrangement value.
- Reject RVC as a default path after poor listening results.
- Keep SoulX-SVC as a backup conversion path only.
- Prioritize true re-singing / vocal regeneration where lyrics, melody, and timbre are rebuilt more cleanly.
- Produce shallow review folders for manual listening.
- Delete failed experiments and temporary work directories.

Current preserved folders:

- `outputs/top_songs`
- `outputs/quality25_trial`
- `outputs/soulx_review`

## Current Baseline

### Existing High-Score Material

`outputs/quality25_trial` contains 3 repaired high-score trial songs:

- `001_95_review_guofeng_月下长安_3bb9c8b78094.wav`
- `002_91_review_sweet_female_Postcards_After_Midnight_781bbc6f14fc.wav`
- `003_76_review_battle_Burn_Through_The_Sky_33a30c367914.wav`

### Existing Backup Result

`outputs/soulx_review` contains 30-second SoulX-SVC preview outputs.

Observed machine vocal artifact scores:

- Track 1: `28 -> 2`
- Track 2: `25 -> 7`
- Track 3: `31 -> 12`

Listening result: slightly better than RVC, but not enough to become the main production path.

## Three Main Tracks

## Track A: SoulX-Singer SVS Re-Singing

Status: first priority.

Purpose:

Use SoulX-Singer as a true singing voice synthesis path instead of audio-to-audio conversion.

Target flow:

```text
high-score song
-> Demucs separate vocals/instrumental
-> extract melody/F0 from existing vocal
-> derive or align lyrics for 30-second section
-> SoulX-Singer SVS generates new vocal
-> remix with instrumental
-> output to shallow review folder
```

Why this is better than SVC:

- SVC inherits too much from the original AI vocal.
- SVS can regenerate vocal texture from lyrics and pitch control.
- If alignment is good, this has a higher ceiling for naturalness.

Implementation tasks:

1. Inspect SoulX SVS CLI requirements:
   - `G:\Cover_vision\SoulX-Singer\cli\inference.py`
   - `G:\Cover_vision\SoulX-Singer\example\infer.sh`
   - `G:\Cover_vision\SoulX-Singer\preprocess\README.md`
2. Build `tools/vocal_repair/soulx_svs_workflow.py`.
3. Generate a 30-second SVS preview for one Chinese high-score song.
4. Generate a 30-second SVS preview for one English high-score song if English metadata path is workable.
5. Output only:
   - `outputs/svs_review/*.wav`
   - `outputs/svs_review/*.txt`
   - `outputs/svs_review/svs_review_manifest.json`
6. Delete work folders unless `--keep-work` is passed.

Acceptance criteria:

- Preview vocal is clearly more human than the original and SoulX-SVC preview.
- Pronunciation is intelligible.
- Pitch follows the original melody closely enough for the backing track.
- No obvious robotic buzz, broken syllables, sudden synthetic high-note cracks.
- The full 30-second preview sounds like one coherent singer, not stitched fragments.

Failure criteria:

- Lyrics are badly misaligned.
- Melody drift makes it unusable over the existing instrumental.
- Vocal timbre is still obviously synthetic or worse than SVC.
- Setup requires manual alignment so heavy that it blocks daily production.

Decision:

- If accepted, make SVS the default high-quality vocal rebuild path.
- If rejected, keep its code but do not output it into final review folders.

## Track B: YingMusic-Singer Melody-Guided Re-Singing

Status: second priority after SoulX-SVS.

Purpose:

Use annotation-free melody guidance to regenerate singing from melody audio and lyrics.

Why this is promising:

- It is designed for melody-guided zero-shot singing.
- It may need less manual MIDI/phoneme alignment than SoulX-SVS.
- It is closer to our desired flow: existing melody -> new human-like vocal.

Current blocker:

`G:\Cover_vision\YingMusic-Singer` exists, but no model checkpoint was found in the local tree.

Implementation tasks:

1. Confirm required checkpoints and expected local paths.
2. Download or locate YingMusic model weights.
3. Validate environment:
   - Python version
   - CUDA compatibility
   - Torch version
   - flash-attention requirement
4. Build `tools/vocal_repair/yingmusic_workflow.py`.
5. Run one 30-second melody-guided preview.
6. Compare against:
   - original repaired vocal
   - SoulX-SVC preview
   - SoulX-SVS preview

Acceptance criteria:

- Better naturalness than SoulX-SVC.
- Less alignment work than SoulX-SVS.
- Stable enough to automate 10-song preview batches.

Failure criteria:

- Checkpoints unavailable or too heavy to run locally.
- Environment setup conflicts with existing ACE-Step / SoulX / RVC stacks.
- Output is unstable, off-pitch, or pronunciation is worse than SoulX-SVS.

Decision:

- If accepted, make YingMusic the preferred melody-guided rebuild path.
- If rejected, keep it as research only and do not retain output folders.

## Track C: DiffSinger / OpenVPI Dedicated Singer

Status: long-term quality ceiling.

Purpose:

Train or adapt a dedicated singing voice model for a fixed account vocal identity.

Why this matters:

- Highest chance of a consistent signature voice.
- Better brand identity for a self-media music account.
- More control than repeated zero-shot conversion.

Cost:

- Requires clean vocal dataset.
- Requires lyrics/pitch alignment.
- Requires training and model management.
- Not the fastest route for tomorrow's iteration.

Implementation tasks:

1. Audit available human vocal references under `human_refs/vocals`.
2. Select only clean, legally usable, high-quality vocal material.
3. Prepare a dataset manifest:
   - audio path
   - language
   - vocal type
   - duration
   - quality notes
4. Decide target voice:
   - Chinese sweet female
   - English pop female
   - battle/anime power female
5. Build dataset preparation scripts only after Track A/B are evaluated.

Acceptance criteria:

- A repeatable voice identity can be created.
- Training quality is audibly better than zero-shot SVC/SVS.
- Dataset prep is not more expensive than generating new songs from scratch.

Failure criteria:

- Human reference data is too noisy or legally unsuitable.
- Alignment workload is too high.
- Training output still sounds synthetic.

Decision:

- Treat as production infrastructure after the short-term rebuild path is chosen.

## Tomorrow Execution Order

1. Preserve current state.
   - Do not delete `outputs/top_songs`.
   - Do not delete `outputs/quality25_trial`.
   - Keep `outputs/soulx_review` as backup comparison.

2. Run Track A: SoulX-SVS 30-second probe.
   - Start with `月下长安`.
   - Use existing lyrics TXT.
   - Use existing separated vocal melody/F0.
   - Produce `outputs/svs_review`.

3. Compare four versions:
   - original repaired preview from `quality25_trial`
   - SoulX-SVC preview from `soulx_review`
   - new SoulX-SVS preview
   - isolated generated vocal when useful

4. If SoulX-SVS is better:
   - Run 3-song 30-second previews.
   - Then run one full song.
   - Only move to `outputs/top_songs` after manual approval.

5. If SoulX-SVS is not better:
   - Keep SoulX-SVC as backup only.
   - Move to Track B and set up YingMusic checkpoint.

6. If both short-term approaches are not enough:
   - Start Track C dataset audit.
   - Stop spending time on conversion-style repair.

## Folder Policy

Final or review folders allowed:

- `outputs/top_songs`
- `outputs/quality25_trial`
- `outputs/soulx_review`
- `outputs/svs_review`
- `outputs/yingmusic_review`

Temporary folders must be deleted by default:

- `outputs/soulx_auto_work`
- `outputs/svs_work`
- `outputs/yingmusic_work`
- `outputs/vocal_resing`
- any failed RVC review folder

No JSON sidecars in final song folders unless needed for a review manifest. For human listening, every review WAV should have a matching TXT.

## Quality Gate

A song can enter final high-quality review only if:

- Machine score is high enough.
- Vocal artifact score improves or stays low.
- Manual listening confirms the vocal sounds closer to human.
- The backing track is not made sharper, thinner, or damaged by the workflow.
- The result has no obvious word breaks, robotic sustained notes, or sudden synthetic noise.

Do not accept a technically improved file if it sounds worse.

## Operational Defaults

- RVC: disabled by default.
- SoulX-SVC: backup preview path only.
- SoulX-SVS: next primary test path.
- YingMusic: second primary test path after checkpoint setup.
- DiffSinger/OpenVPI: long-term dedicated singer path.

Default preview length:

- `30s` for first evaluation.

Default full-song rule:

- Run full song only after a 30-second preview is accepted.

Default cleanup:

- Delete temporary work folders after every run.

