Python SuperSmoother
====================

This is an efficient implementation of Friedman's SuperSmoother [1]
algorithm in pure Python. It makes use of [numpy](http://numpy.org)
for fast numerical computation.

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
You can see an example of the code in action [on nbviewer](http://nbviewer.ipython.org/github/jakevdp/supersmoother/blob/master/examples/Supersmoother.ipynb)

Testing
-------
This code has full unit tests implemented in [nose](https://nose.readthedocs.org/en/latest/). With ``nose`` installed, you can run the test suite using
```
$ nosetests supersmoother
```

[1]: Friedman, J. H. (1984) A variable span scatterplot smoother. Laboratory for Computational Statistics, Stanford University Technical Report No. 5. ([pdf](http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-3477.pdf))

