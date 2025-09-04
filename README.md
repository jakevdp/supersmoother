# Python SuperSmoother

This is an efficient implementation of Friedman's SuperSmoother [1]
algorithm in pure Python. It makes use of [NumPy](http://numpy.org)
for fast numerical computation.

[![DOI](https://zenodo.org/badge/9372/jakevdp/supersmoother.svg)](http://dx.doi.org/10.5281/zenodo.14475)
[![version status](http://img.shields.io/pypi/v/supersmoother.svg?style=flat)](https://pypi.python.org/pypi/supersmoother)
[![build status](https://github.com/jakevdp/supersmoother/actions/workflows/test.yml/badge.svg)](https://github.com/jakevdp/supersmoother/actions/workflows/test.yml)
[![license](http://img.shields.io/badge/license-BSD-blue.svg?style=flat)](https://github.com/jakevdp/supersmoother/blob/main/LICENSE)

## Installation
To install the released version, type
```
$ pip install supersmoother
```

This will also install `numpy` if not already installed.

To install the bleeding-edge source, download the source code from http://github.com/jakevdp/supersmoother and type:
```
$ pip install .
```

The only package dependency is `numpy`; `scipy` and `pytest` are also required if you wish to run the test suite.

## Example
The package includes several example notebooks showing the code in action.
You can see these in the `examples/` directory, or view them statically
[on nbviewer](http://nbviewer.ipython.org/github/jakevdp/supersmoother/blob/main/examples/Index.ipynb)

## Testing
This code has full unit tests implemented using [pytest](https://pytest.org).
They can be run as follows from the source directory:
```
$ pip install .[dev]
$ pytest -n auto supersmoother
```
The package is tested with Python versions 3.9 through 3.14.

## Authors
``supersmoother`` was created by [Jake VanderPlas](http://vanderplas.com)

## Citing This Work
If you use this code in an academic publication, please consider including a citation to our work.
Citation information in a variety of formats can be found [on zenodo](http://dx.doi.org/10.5281/zenodo.14475).

## References
[1] Friedman, J. H. (1984) A variable span scatterplot smoother. Laboratory for Computational Statistics, Stanford University Technical Report No. 5. ([pdf](http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-3477.pdf))
