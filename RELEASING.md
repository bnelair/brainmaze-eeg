# Releasing `brainmaze-eeg`

This package follows the BrainMaze family release process. The full guide (how a release
works, recovering from a failed release, one-time setup, branch protection, dependency order,
compatibility policy, troubleshooting) is
**[RELEASING.md in bnelair/brainmaze-sphinx](https://github.com/bnelair/brainmaze-sphinx/blob/main/RELEASING.md)**.

## Quick steps

1. Make sure `main` has what you want to release and CI is green.
2. **Actions → Prepare release → Run workflow**, pick `patch` / `minor` / `major`
   (changed numerical results count as breaking).
3. Review the bot's **"Release vX.Y.Z"** PR and check that its diff is exactly the one
   `version = "X.Y.Z"` line in `pyproject.toml`. CI and the Version guard don't run on it
   (PRs opened with `GITHUB_TOKEN` trigger no workflows), so this check is yours. Then
   squash-merge it.
4. The merge triggers the **Release** workflow automatically (it has no manual trigger). It
   tests, builds, publishes to [PyPI](https://pypi.org/project/brainmaze-eeg/) with Trusted
   Publishing, pushes tag `vX.Y.Z` and creates the GitHub Release. Watch it under
   *Actions → Release*.

If the Release run fails part-way, **don't wait for the next push to `main`** (that would
publish and tag a different commit) and **don't run Prepare release again** (that would skip
the version). Re-run the failed jobs on that run, or finish by hand; see
[Recovering from a failed release](https://github.com/bnelair/brainmaze-sphinx/blob/main/RELEASING.md#recovering-from-a-failed-release).

Never edit `[project].version` in a normal PR. The **Version guard** check flags it; the
check is advisory (no ruleset requires it), so reviewers must not merge a PR it flags.

## This repository

| | |
|---|---|
| PyPI | [`brainmaze-eeg`](https://pypi.org/project/brainmaze-eeg/) |
| import | `import brainmaze_eeg` (`brainmaze_eeg.__version__` comes from installed metadata) |
| thin callers of [brainmaze-sphinx](https://github.com/bnelair/brainmaze-sphinx) | `ci.yml` (tests), `docs.yml` (GitHub Pages), `prepare-release.yml` |
| with local logic | `release.yml`: calls the shared guard + test + build, then its own `publish` job uploads to PyPI, pushes the tag and creates the GitHub Release. `version-guard.yml`: entirely local. |
| PyPI Trusted Publisher | owner `bnelair`, repo `brainmaze-eeg`, workflow `release.yml`, no environment |
