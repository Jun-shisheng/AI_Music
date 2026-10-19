# Contributing to AI Music Studio

Thanks for helping. This project is local-first and measurement-first: a change
is welcome when it is small, reproducible, and backed by a test or a number.

By contributing you agree that your work is licensed under the repository's MIT
license.

## Before you start

- Open an issue for anything larger than a bug fix or a docs typo. Describe the
  problem, the environment, and how to reproduce it. For audio defects, include
  the measured numbers (`arc_qa.py`, `audio_qa.py`) and the exact seconds that
  sound wrong.
- Small fixes may go straight to a pull request.
- Never commit `.env`, model weights, generated audio, `outputs/`, `inputs/`,
  `logs/`, `data/`, or anything else on `.gitignore`.

## Branch model

Two long-lived branches:

| Branch | Role |
|---|---|
| `main` | release branch; every commit is a released state, tagged `vX.Y.Z` |
| `develop` | integration branch; the target for all feature and fix work |

```
develop ──●──●──●──●──●──►   integration; CI on every push
         /     \       \
feat/x  ●──●    ●──●    release PR ──► main ──► tag vX.Y.Z
                                             \
                                              back-merge into develop
```

Short-lived branches:

| Prefix | Base | Merges into |
|---|---|---|
| `feat/<slug>` | `develop` | `develop` |
| `fix/<slug>` | `develop` | `develop` |
| `docs/<slug>` | `develop` | `develop` |
| `test/<slug>` | `develop` | `develop` |
| `refactor/<slug>` | `develop` | `develop` |
| `chore/<slug>` | `develop` | `develop` |
| `release/vX.Y.Z` | `develop` | `main` (optional stabilization branch) |
| `hotfix/<slug>` | `main` | `main`, then back-merge into `develop` |

Rules:

1. Never commit directly to `main` or `develop`; both are protected.
2. Feature, fix, docs, test, refactor and chore branches start from `develop`
   and open a pull request back to `develop`.
3. Rebase on the latest `develop` before opening or updating a pull request.
4. Keep one topic per branch and one topic per pull request; delete the branch
   after merge.
5. External contributors fork the repository, push the branch to their fork and
   open a pull request against `AI_Music:develop`.
6. `develop` must stay green. A red CI on `develop` is a stop-the-line event.
7. `main` only receives release and hotfix pull requests. Do not develop on it.

### Releases

1. Move the relevant entries from `[Unreleased]` to a new version section in
   `CHANGELOG.md` on `develop`.
2. Optionally create `release/vX.Y.Z` from `develop` for final stabilization.
3. Open a pull request from `develop` (or the release branch) to `main`.
4. Squash-merge when CI is green, then tag `main`: `git tag -a vX.Y.Z`.
5. Publish the GitHub release from the tag.
6. Back-merge `main` into `develop` with a merge commit so the release commit
   and tag are in the integration history. Do not squash the back-merge.

Releases are annotated tags on `main` using `v0.x.y` (semantic versioning). A
release note lists user-visible changes and any API-contract change.

### Hotfixes

Branch `hotfix/<slug>` from `main`, open a pull request to `main`, tag a patch
release, then back-merge `main` into `develop` with a merge commit.

### Historical branches

`master` is a frozen July 2026 snapshot. It is more than 40 commits behind
`main` and receives no changes. Use `main`/`develop` for all work; do not target
pull requests at `master`.

### Branch protection

Recommended settings (maintainers configure these in the GitHub repository
settings):

`main`

- require a pull request before merging;
- require the CI checks: `backend (ubuntu-latest)`,
  `backend (windows-latest)` and `frontend`;
- require conversation resolution;
- require linear history;
- disallow force pushes and branch deletion.

`develop`

- require a pull request before merging;
- require the same CI checks;
- require conversation resolution;
- disallow force pushes and branch deletion;
- do **not** require linear history: release back-merges use a merge commit.

Keep **enforce for administrators** off while the project has a single
maintainer, so the owner can recover from a broken state. Turn it on once there
are two active maintainers; then also require one approving review on both
branches.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request targeting
`main` or `develop`:

- backend tests on Ubuntu and Windows with Python 3.11;
- `npm ci && npm run build` for the frontend.

A red CI run is a blocker: fix it in the same branch rather than merging with a
failing required check. Dependabot opens monthly dependency PRs using the same
CI and review rules; security updates are separate and remain enabled.

Version updates target `develop`. Dependabot security updates follow the
repository's default branch (`main`); merge them there, then back-merge `main`
into `develop` so the fix is not lost on the integration branch.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <imperative summary>

<why this change is needed, and what it changes>

<how it was verified>
```

- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`,
  `chore`.
- Subject line at most 72 characters, no trailing period, present tense.
- One logical change per commit where practical. Squash fixup commits before
  review.
- Reference issues in the body (`Refs #12`) or with a closing keyword
  (`Closes #12`).

Examples:

```
feat(api): add SSE job progress
fix(queue): claim jobs atomically on restart
docs(readme): document 8 GB vocal artifacts
test(audio): cover path traversal in /audio
```

## Issues

Search the existing issues and Discussions before opening a new one. Use the
provided templates; an issue with a reproduction is worth more than a long
description without one.

### Bug reports

- Use the **Bug report** template and include the OS, GPU/VRAM, Python version,
  commit SHA and backend.
- Give a minimal reproduction and state the expected and actual result.
- For audio defects, include the exact seconds that sound wrong and the output
  of `audio_qa.py` / `arc_qa.py`. A number is actionable; "it sounds bad" is not.
- Redact API keys, usernames and absolute personal paths.

### Feature requests

- Use the **Feature request** template and lead with the problem, not only the
  proposed solution.
- State how we could tell it works: a test, a measurement, or a concrete
  acceptance check.
- Large or breaking changes should start as an RFC issue before any code.

### Questions and security

- Usage questions and design discussions belong in GitHub Discussions.
- Security vulnerabilities go through [SECURITY.md](SECURITY.md), never a public
  issue.

### Triage

Maintainers apply one or more labels: `bug`, `enhancement`, `documentation`,
`needs reproduction`, `area: api`, `area: frontend`, `area: tools`,
`breaking change`, `dependencies`. An issue is ready for work when it has a
reproduction (bugs) or an acceptance check (features). `good first issue` marks
a self-contained task with clear acceptance criteria and no hidden design work.

## Pull requests

Open feature and fix pull requests against `develop`. Release and hotfix pull
requests target `main`. Use a draft PR while the branch is still moving, and
mark it ready only when CI is green and the description is complete. The PR
title follows Conventional Commits, because it becomes the squash-merge
subject: `fix(queue): claim jobs atomically on restart`.

Use the pull-request template. A pull request is ready for review when:

- [ ] `cd services/api && python -m pytest -q` passes.
- [ ] `cd apps/web && npm run build` passes.
- [ ] The change has a test, or the reason it cannot be tested is explained.
- [ ] `docs/api.md` is updated when the API contract changes.
- [ ] UI changes include a before/after screenshot.
- [ ] No secrets, weights, audio, logs, or generated data are included.
- [ ] Commits follow the convention above and the branch is rebased on its
      target (`develop` for feature work, `main` for release/hotfix PRs).

Review and merge:

- One maintainer review is required; more than one for API or queue changes.
- The author responds to every comment and resolves the conversation after the
  change is pushed. Do not force-push a branch that is under review unless the
  reviewer asks for a rebase.
- CI must be green at merge time. Do not merge with a failing or missing
  required check.
- The maintainer squash-merges using the PR title, then deletes the branch.
- Aim for a focused diff under roughly 400 changed lines. Split a larger change
  into reviewable pull requests.
- A maintainer who is also the author may self-merge only after CI is green and
  the PR has been open long enough for comment; once a second maintainer
  exists, that person's review is required instead.

For a large or breaking change, open an RFC issue first and agree on the shape
before writing code.

## Releases

1. Move the relevant entries from `[Unreleased]` to a new version section in
   `CHANGELOG.md`.
2. Tag the release on `main`: `git tag -a v0.X.Y -m "v0.X.Y"`.
3. Push the tag: `git push origin v0.X.Y`.
4. Publish the GitHub release from the tag, using the changelog entry as the
   notes.

## Code style

- Python 3.11+, type hints on public functions, and comments that explain
  *why* rather than restating the code.
- Match the surrounding code; the project does not enforce a formatter yet.
  `ruff` and `pyright` are planned for CI.
- Frontend: TypeScript strict mode, Tailwind utility classes, and the existing
  design tokens in `apps/web/src/app/globals.css`. Avoid a new runtime
  dependency unless it clearly earns its weight.
- Tests live next to the layer they cover. API tests use pytest plus the ASGI
  transport; queue tests stub the generator and never load a model.

## Security

- The services bind to `127.0.0.1` by design. Do not change that without a
  discussion.
- Report vulnerabilities privately through GitHub Security Advisories rather
  than in a public issue. See [SECURITY.md](SECURITY.md).
- Redact API keys, usernames, and absolute paths from logs pasted into issues.
