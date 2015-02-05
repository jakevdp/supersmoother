Python SuperSmoother
====================

This is an efficient implementation of Friedman's SuperSmoother [1]
algorithm in pure Python. It makes use of [numpy](http://numpy.org)
for fast numerical computation.

[![DOI](https://zenodo.org/badge/9372/jakevdp/supersmoother.svg)](http://dx.doi.org/10.5281/zenodo.14475)
[![version status](https://pypip.in/v/supersmoother/badge.png)](https://pypi.python.org/pypi/supersmoother)
[![downloads](https://pypip.in/d/supersmoother/badge.png)](https://pypi.python.org/pypi/supersmoother)
[![build status](https://travis-ci.org/jakevdp/supersmoother.png?branch=master)](https://travis-ci.org/jakevdp/supersmoother)

Installation
------------
Installation is simple: To install the released version, type

    $ pip install supersmoother

To install the bleeding-edge source, download the source code from http://github.com/jakevdp/supersmoother and type:

    $ python setup.py install

The only package dependency is ``numpy``; ``scipy`` is also required if you want to run the unit tests.

Example
-------
The package includes several example notebooks showing the code in action.
You can see these in the ``examples/`` directory, or view them statically
[on nbviewer](http://nbviewer.ipython.org/github/jakevdp/supersmoother/blob/master/examples/Index.ipynb)

Testing
-------
This code has full unit tests implemented in [nose](https://nose.readthedocs.org/en/latest/). With ``nose`` installed, you can run the test suite using
```
$ nosetests supersmoother
```

Authors
-------
``supersmoother`` was created by [Jake VanderPlas](http://vanderplas.com)

Citing This Work
----------------
If you use this code in an academic publication, please consider including a citation to our work.
Citation information in a variety of formats can be found [on zenodo](http://dx.doi.org/10.5281/zenodo.14475).

References
----------
[1] Friedman, J. H. (1984) A variable span scatterplot smoother. Laboratory for Computational Statistics, Stanford University Technical Report No. 5. ([pdf](http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-3477.pdf))

