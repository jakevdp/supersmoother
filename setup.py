import io
import os
import re

from setuptools import setup


def read(path, encoding='utf-8'):
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()


def version(path):
    """Obtain the packge version from a python file e.g. pkg/__init__.py

    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    version_file = read(path)
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


DESCRIPTION = "Python implementation of Friedman's Supersmoother"
LONG_DESCRIPTION = """
SuperSmoother in Python
=======================
This is an efficient implementation of Friedman's SuperSmoother based in
Python. It makes use of numpy for fast numerical computation.

For more information, see the github project page:
http://github.com/jakevdp/supersmoother
"""
NAME = "supersmoother"
AUTHOR = "Jake VanderPlas"
AUTHOR_EMAIL = "jakevdp@uw.edu"
MAINTAINER = "Jake VanderPlas"
MAINTAINER_EMAIL = "jakevdp@uw.edu"
URL = 'http://github.com/jakevdp/supersmoother'
DOWNLOAD_URL = 'http://github.com/jakevdp/supersmoother'

VERSION = version('supersmoother/__init__.py')

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      install_requires=["numpy"],
      tests_require=["scipy"],
      extras_require={
          "dev": [
              "pytest",
              "pytest-xdist",
              "scipy"
          ]
      },
      packages=['supersmoother',
                'supersmoother.tests',
            ],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
      ],
     )
