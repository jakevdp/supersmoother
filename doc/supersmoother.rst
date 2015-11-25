.. _supersmoother

Supersmoother: Adaptively-tuned Smoothing
=========================================

The Supersmoother algorithm is essentially an adaptive, variable-span linear
smoother. Consider the following data, in which the period, amplitude, and
(known) uncertainty of the data vary as a function of *x*:

.. ipython::

    In [1]: import numpy as np

    In [2]: rng = np.random.RandomState(0)
    
    In [3]: x = 10 * rng.rand(200)
    
    In [4]: y_err = 0.1 * x
    
    In [5]: y = np.sin(2 * np.pi * (1 - 0.1 * x) ** 2) + y_err * rng.randn(len(x))

We can visualize this data as follows:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn; seaborn.set()

    rng = np.random.RandomState(0)
    x = 10 * rng.rand(200)
    y_err = 0.1 * x
    y = np.sin(2 * np.pi * (1 - 0.1 * x) ** 2) + y_err * rng.randn(len(x))
    plt.errorbar(x, y, y_err, fmt='o', color='gray', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-3, 3)

If we try a simple linear smoother on this data, we find that the best choice
of span seems to vary across the dataset:

.. plot::

    from supersmoother import LinearSmoother
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn; seaborn.set()

    rng = np.random.RandomState(0)
    x = 10 * rng.rand(200)
    y_err = 0.1 * x
    y = np.sin(2 * np.pi * (1 - 0.1 * x) ** 2) + y_err * rng.randn(len(x))
    plt.errorbar(x, y, y_err, fmt='o', color='gray', alpha=0.3)

    xfit = np.linspace(0, 10, 1000)

    for span in [0.05, 0.2, 0.5]:
        smoother = LinearSmoother(span)
        smoother.fit(x, y, y_err)
        plt.plot(xfit, smoother.predict(xfit),
                 label="LinearSmoother(span={0:.2f})".format(span))
    plt.legend(loc='lower left')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-3, 3)

The short span seems appropriate for the left side of the plot, while the long
span seems more appropriate for the right side of the plot.


Supersmoother Example
---------------------
The :class:`~supersmoother.SuperSmoother` algorithm provides an
automatically-tuned adaptive-span smoother for use with such data.
The interface is exceedingly simple:

.. ipython::

    In [1]: from supersmoother import SuperSmoother

    In [2]: smoother = SuperSmoother()
    
    In [3]: smoother.fit(x, y, y_err)
    
    In [4]: y_smooth = smoother.predict(x)

The result is visualized here:

.. plot::

    from supersmoother import SuperSmoother
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn; seaborn.set()

    rng = np.random.RandomState(0)
    x = 10 * rng.rand(200)
    y_err = 0.1 * x
    y = np.sin(2 * np.pi * (1 - 0.1 * x) ** 2) + y_err * rng.randn(len(x))
    plt.errorbar(x, y, y_err, fmt='o', color='gray', alpha=0.3)

    xfit = np.linspace(0, 10, 1000)

    smoother = SuperSmoother()
    smoother.fit(x, y, y_err)
    plt.plot(xfit, smoother.predict(xfit))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-3, 3)


Understanding the SuperSmoother
-------------------------------
The supersmoother algorithm is fairly straightforward to understand. It is
a procedural recipe:

1. Compute three simple linear smooths with spans of 0.05, 0.2, and 0.5,
   along with the cross-validated residuals of the smooths.

2. Smooth these residuals using the midrange span (i.e. span=0.2)

3. For each point, select the span yielding the best smoothed residual

4. Smooth these *span values* as a function of *x*, using the midrange span.

5. Compute the final smooth with a linear smoother based on these smoothed
   span values.

We can visualize these steps using the attributes of the
:class:`~supersmoother.SuperSmoother` class. First we can plot and compare
the three component smooths:

.. plot::

    from supersmoother import SuperSmoother
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn; seaborn.set()

    rng = np.random.RandomState(0)
    x = 10 * rng.rand(200)
    y_err = 0.1 * x
    y = np.sin(2 * np.pi * (1 - 0.1 * x) ** 2) + y_err * rng.randn(len(x))
    plt.errorbar(x, y, y_err, fmt='o', color='gray', alpha=0.3)

    xfit = np.linspace(0, 10, 1000)

    smoother = SuperSmoother()
    smoother.fit(x, y, y_err)

    for component in smoother.primary_smooths:
        plt.plot(xfit, component.predict(xfit),
                 label='span = {0:.2f}'.format(component.span))

    plt.legend(loc='lower left');

    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-3, 3)

From these initial fits, we arrive at a smoothed optimal span value as a
function of *x*, and use this span value for the local smoothing of the data:

.. plot::

    from supersmoother import SuperSmoother
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn; seaborn.set()

    rng = np.random.RandomState(0)
    x = 10 * rng.rand(200)
    y_err = 0.1 * x
    y = np.sin(2 * np.pi * (1 - 0.1 * x) ** 2) + y_err * rng.randn(len(x))
    
    xfit = np.linspace(0, 10, 1000)

    smoother = SuperSmoother()
    smoother.fit(x, y, y_err)
    
    gs = plt.GridSpec(4, 1, hspace=0.15)
    fig = plt.figure()
    ax0 = fig.add_subplot(gs[:-1])
    ax0.xaxis.set_major_formatter(plt.NullFormatter())
    ax1 = fig.add_subplot(gs[-1])
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    
    ax0.errorbar(x, y, y_err, fmt='o', color='gray', alpha=0.3)
    ax0.plot(xfit, smoother.predict(xfit))
    ax0.set_ylabel('y')
    ax0.set_ylim(-2.5, 2.5)
    
    ax1.plot(xfit, smoother.span(xfit), '-k', alpha=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('span')
    ax1.set_ylim(0, 0.5)

By adjusting the span based on the residuals of the smooth, we find a near optimal smoothing at every *x* location.
