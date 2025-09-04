# How to Release

Here's a quick step-by-step for cutting a new release of supersmoother.

## Pre-release

1. update version in `supersmoother/__init__.py`

2. create a release tag; e.g.
   ```
   $ git tag -a v0.5.0 -m 'version 0.5.0 release'
   ```

3. push the commits and tag to github

4. confirm that CI tests pass on github

5. under "tags" on github, update the release notes

6. build the release and push to PyPI:
   ```
   $ pip install build twine
   $ rm -r dist
   $ python -m build
   $ ls dist/  # double check that the version is correct
   supersmoother-0.5.0-py3-none-any.whl	supersmoother-0.5.0.tar.gz
   $ twine upload dist/*
   ```

## Post-release

update version in `supersmoother/__init__.py` to next development version;
e.g. '0.6.0.dev0'
