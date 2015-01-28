"""Python implementation of Friedman's Supersmoother"""
from __future__ import absolute_import

__version__ = '0.3.1'

from .smoother import MovingAverageSmoother, LinearSmoother
from .supersmoother import SuperSmoother
