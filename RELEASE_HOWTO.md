# How to Release

Here's a quick step-by-step for cutting a new release of gatspy.

## Pre-release

1. update version in ``supersmoother.__init__.py``

2. create a release tag; e.g.
   ```
   $ git tag -a v0.2 -m 'version 0.2 release'
   ```

3. push the commits and tag to github

4. confirm that CI tests pass on github

5. under "tags" on github, update the release notes

6. push the new release to PyPI:
   ```
   $ python setup.py sdist upload
   ```

## Post-release

1. update version in ``gatspy.__version__`` to next version; e.g. '0.3.dev'