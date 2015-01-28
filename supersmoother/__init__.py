"""Python implementation of Friedman's Supersmoother"""
from __future__ import absolute_import

__version__ = '0.3.1-git'

from .smoother import MovingAverageSmoother, LinearSmoother
from .supersmoother import SuperSmoother
