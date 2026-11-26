# Releasing py-parsec

py-parsec is published on PyPI as `python-parsec` (PyPI refuses `py-parsec`
as too similar to the unrelated `pyparsec`), as a source distribution only:
the extension modules have to be compiled against the user's PaRSEC, so
there are no wheels.

## One-time setup

1. On PyPI, add a trusted publisher for the `python-parsec` project (pending
   publisher for a first release): owner `QingleiCao`, repository
   `py-parsec`, workflow `release.yml`, environment `pypi`.
2. In the GitHub repository settings, create an environment named `pypi`
   (optionally with required reviewers).

## Each release

1. Make sure CI passes on `main`.
2. Set the version in `pyproject.toml` and move the `Unreleased` entries of
   `CHANGELOG.md` under the new version with its date.
3. Commit, then tag and push: `git tag v0.1.0 && git push origin v0.1.0`.
4. The `release` workflow builds the sdist, checks it, builds and tests it
   against PaRSEC, publishes it to PyPI and attaches it to a GitHub release.
