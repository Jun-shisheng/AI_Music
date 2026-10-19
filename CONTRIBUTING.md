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

The repository uses **GitHub Flow**. There is no long-lived `develop` branch:
`main` is the release branch and should always build and pass tests.

1. Branch from the latest `main`; do not commit directly to `main`.
2. Use a short-lived branch with a typed name:

   | Prefix | Use |
   |---|---|
   | `feat/<slug>` | new user-visible capability |
   | `fix/<slug>` | bug fix |
   | `docs/<slug>` | documentation only |
   | `test/<slug>` | tests only |
   | `refactor/<slug>` | no behavior change |
   | `chore/<slug>` | tooling, dependencies, housekeeping |

3. Keep one topic per branch and one topic per pull request.
4. Rebase on `main` before opening or updating a pull request.
5. If you do not have write access, fork the repository, push the branch to
   your fork, and open a pull request against `AI_Music:main`.
6. Maintainers follow the same flow. `main` is protected: changes land through
   a pull request, and force-pushing to `main` is not allowed.

Releases are annotated tags on `main` using `v0.x.y` (semantic versioning). A
release note lists user-visible changes and any API-contract change.

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

## Pull requests

Use the pull-request template. A pull request is ready for review when:

- [ ] `cd services/api && python -m pytest -q` passes.
- [ ] `cd apps/web && npm run build` passes.
- [ ] The change has a test, or the reason it cannot be tested is explained.
- [ ] `docs/api.md` is updated when the API contract changes.
- [ ] UI changes include a before/after screenshot.
- [ ] No secrets, weights, audio, logs, or generated data are included.
- [ ] Commits follow the convention above and the branch is rebased on `main`.

Review and merge:

- One maintainer review is required; more than one for API or queue changes.
- Address review comments with new commits, then squash-merge.
- Delete the branch after merge.

For a large or breaking change, open an RFC issue first and agree on the shape
before writing code.

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
