## What & why

<!-- What does this change and why? Link any related issue. -->

## Checklist

- [ ] Tests added/updated for the behavior change
- [ ] `pytest` passes locally — the coverage floor in `pyproject.toml`'s `[tool.coverage.report] fail_under` is a ratchet, raised as coverage improves and never lowered to make a change pass
- [ ] `pyright` passes (basic mode, pinned to the 3.9 floor — see `.github/workflows/test.yml`'s `pyright` job)
- [ ] `ruff check src/ tests/` is clean for files touched
- [ ] No secrets, credentials, or proprietary prompt/logic content committed
- [ ] Public API is additive-only (no renames or removals without a documented migration + consumer sweep — this package is pinned by ~10 repos)
- [ ] Does NOT touch the `version` field in `pyproject.toml` / `src/nousergon_lib/__init__.py`, and does not modify `.github/workflows/auto-version-bump.yml` — version bumps are automated on merge to `main`, never hand-edited in a feature PR

## Test plan

<!-- How you verified this works. -->

---

**Prepared by:** <!-- model name from the session prompt -->
