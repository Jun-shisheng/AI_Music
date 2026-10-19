# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Planned

- Pluggable generation backends behind a documented adapter protocol
- Open recipe format with example recipes
- SSE job progress and a waveform player
- Reproducible benchmark harness
- LoRA training workflow (code only)

## [0.1.0] - 2026-10-19

First public snapshot of the local studio and the measurement toolkit.

### Added

- Measured QA toolkit: `arc_qa`, `diction_qa`, `audio_qa`, `loudness`,
  `export_delivery`, `battle_review`.
- FastAPI studio API with arrange, generate, jobs, library and audio endpoints.
- Next.js studio: arrangement, professional create, game-asset workbench and
  library, with a new design system and responsive layout.
- `generation_type` contract for `song`, `bgm_loop` and `sfx`.
- Queue restart recovery, atomic job claiming and empty-output detection.
- Contribution workflow: `CONTRIBUTING.md`, PR and issue templates,
  `CODEOWNERS`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, CI and Dependabot.
- GitHub Actions CI: backend tests on Ubuntu and Windows, frontend build.
- README sections for the measured engine comparison, hardware limits, the
  8 GB laptop vocal-noise problem, roadmap and FAQ.

### Fixed

- `/audio/{filename}` could serve arbitrary files through an absolute path or
  `..` segments; it now serves only direct `.wav` children of `OUTPUT_DIR`.
- The default generation backend could never run; the ACE-Step REST backend is
  now the default and `subprocess` fails with a clear message.
- Jobs interrupted by a restart stayed `running` forever.
- `/jobs` and `/library` now report the resolved instrumental/loop/duration
  values used for generation.
- The create-page placeholder rendered `&#10;` literally.
- The asset workbench reset the chosen duration on every keystroke.
- The library did not start playback when switching tracks.
- Short takes crashed `arc_qa`; mono stems crashed `transition_repair`; a
  missing `inputs/` directory crashed delivery packaging.

### Security

- Confined audio serving to the output directory (path traversal).
- Documented the single-user, localhost-only security model in `SECURITY.md`.
