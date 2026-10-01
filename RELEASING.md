# Releasing `brainmaze-eeg`

This package follows the BrainMaze family release process. The full guide (one-time setup,
dependency order, compatibility policy, troubleshooting) is in
**[bnelair/brainmaze RELEASING.md](https://github.com/bnelair/brainmaze/blob/main/RELEASING.md)**.

## Quick steps

1. Make sure `main` has what you want to release and CI is green.
2. **Actions → Prepare release → Run workflow**, pick `patch` / `minor` / `major`
   (changed numerical results count as breaking).
3. Review and squash-merge the bot's **"Release vX.Y.Z"** PR (it only changes the
   `version =` line in `pyproject.toml`).
4. **Actions → Release** then tests, builds, publishes to
   [PyPI](https://pypi.org/project/brainmaze-eeg/) with Trusted Publishing, pushes tag `vX.Y.Z`
   and creates the GitHub Release.

Never edit `version =` in a normal PR; the **Version guard** check rejects it.

## This repository

| | |
|---|---|
| PyPI | [`brainmaze-eeg`](https://pypi.org/project/brainmaze-eeg/) |
| import | `import brainmaze_eeg` (`brainmaze_eeg.__version__` comes from installed metadata) |
| workflows | `ci.yml` (tests), `docs.yml` (GitHub Pages), `prepare-release.yml`, `release.yml`, `version-guard.yml`, all thin callers of [brainmaze-sphinx](https://github.com/bnelair/brainmaze-sphinx) |
| PyPI Trusted Publisher | owner `bnelair`, repo `brainmaze-eeg`, workflow `release.yml`, no environment |
