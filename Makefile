test:
	nosetests supersmoother

doctest:
	nosetests --with-doctest supersmoother

test-coverage:
	nosetests --with-coverage --cover-package=supersmoother

test-coverage-html:
	nosetests --with-coverage --cover-html --cover-package=supersmoother
