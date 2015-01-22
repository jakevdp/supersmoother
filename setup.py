from distutils.core import setup

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
LICENSE = 'BSD 3-clause'

import supersmoother
VERSION = supersmoother.__version__

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
      license=LICENSE,
      packages=['supersmoother',
                'supersmoother.tests',
            ],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4'],
     )
