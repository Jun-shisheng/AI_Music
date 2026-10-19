## What changed

<!-- One paragraph. Link the issue this closes. -->

Closes #

## Why

<!-- The user-visible problem or the measured failure this fixes. -->

## How it was verified

- [ ] `cd services/api && python -m pytest -q`
- [ ] `cd apps/web && npm run build`
- [ ] Manual check (describe it, or attach a screenshot for UI changes)

## Checklist

- [ ] One topic in this pull request; branch is rebased on `main`
- [ ] Tests added or updated for the behavior change
- [ ] `docs/api.md` updated if the API contract changed
- [ ] No `.env`, model weights, audio, `outputs/`, `inputs/`, or logs included
- [ ] Commits follow the Conventional Commits format
