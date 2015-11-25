.. _basic_smoothers

Basic Neighborhood-Based Smoothing
==================================

For the sake of data visualization, it is often nice to be able to draw a
smooth curve through a noisy scatter-plot. For example, consider the following
data:

.. ipython::

    In [1]: import numpy as np

    In [2]: rng = np.random.RandomState(0)
    
    In [3]: x = np.sort(10 * rng.rand(100))
    
    In [4]: y = np.sin(2 * x) + 0.4 * rng.randn(100)

here is a visualization of this data:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()  # use seaborn styles

    rng = np.random.RandomState(0)
    x = np.sort(10 * rng.rand(100))
    y = np.sin(2 * x) + 0.4 * rng.randn(100)
    plt.plot(x, y, 'o', color='gray', alpha=0.7);

How might we draw a smooth curve through it?

Moving Average Smoothing
------------------------
As a simple starting-point, we might think about drawing a moving average of
the datapoints: for each candidate *x* value, we can take the mean of the *y*
values of the nearby points.
This style of smoothing is implemented in
:class:`~supersmoother.MovingAverageSmoother`, and can be used as follows,
where we will set the *span* of the smoother to be 1/10 of the dataset:

.. ipython::

    In [1]: from supersmoother import MovingAverageSmoother

    In [2]: smoother = MovingAverageSmoother(span=0.1)

    In [3]: smoother.fit(x, y);

    In [4]: y_smooth = smoother.predict(x)

Applying this smoother to the data yields the following smooth:

.. plot::

    from supersmoother import MovingAverageSmoother

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()  # use seaborn styles

    rng = np.random.RandomState(0)
    x = np.sort(10 * rng.rand(100))
    y = np.sin(2 * x) + 0.4 * rng.randn(100)

    model = MovingAverageSmoother(0.1)
    model.fit(x, y)

    xfit = np.linspace(0, 10, 1000)
    plt.plot(x, y, 'o', color='gray', alpha=0.3)
    plt.plot(xfit, model.predict(xfit));

Of course, the resulting smooth depends highly on the span that you choose:
an increasing span size leads to more smoothing:


.. plot::

    from supersmoother import MovingAverageSmoother

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()  # use seaborn styles

    rng = np.random.RandomState(0)
    x = np.sort(10 * rng.rand(100))
    y = np.sin(2 * x) + 0.4 * rng.randn(100)

    plt.plot(x, y, 'o', color='gray', alpha=0.3)
    xfit = np.linspace(0, 10, 1000)

    for span in [0.05, 0.2, 0.5]:
        smoother = MovingAverageSmoother(span)
        smoother.fit(x, y)
        plt.plot(xfit, smoother.predict(xfit),
                 label="MovingAverageSmoother(span={0:.2f})".format(span))
    plt.legend()

Evidently, we need to choose the span value very carefully!


Local Linear Smoothing
----------------------
One weakness of the moving average smoothing scheme is that in locations where
the data is sparse, the "smooth" curve displays flat plateaus. It would be nice
to reduce this by allowing the local model to adapt to any aggregate trends
within the data. For example, rather than simply taking the average of each
local group, we might compute a line of best fit within each local group.

Such an approach is implemented in the :class:`~supersmoother.LinearSmoother`, and can be used as follows:

.. ipython::

    In [1]: from supersmoother import LinearSmoother

    In [2]: smoother = MovingAverageSmoother(span=0.1)

    In [3]: smoother.fit(x, y);

    In [4]: y_smooth = smoother.predict(x)

Notice that the interface here is identical to that of :class:`~supersmoother.MovingAverageSmoother`, above.
We can compare the output of the two smoothing approaches for the same data:

.. plot::

    from supersmoother import MovingAverageSmoother, LinearSmoother

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()  # use seaborn styles

    rng = np.random.RandomState(0)
    x = np.sort(10 * rng.rand(100))
    y = np.sin(2 * x) + 0.4 * rng.randn(100)
    xfit = np.linspace(0, 10, 1000)
    
    plt.plot(x, y, 'o', color='gray', alpha=0.3)

    for Model in [MovingAverageSmoother, LinearSmoother]:
        model = Model(0.1)
        model.fit(x, y)

        plt.plot(xfit, model.predict(xfit), '-', label=Model.__name__ + '(0.1)')
    plt.legend()

As we can see, the linear smoother effectively interpolates across the
flat regions created by the moving average smoother.
Still, just as with the simpler smoother above, the results are highly
dependent on the choice of span:

.. plot::

    from supersmoother import LinearSmoother

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()  # use seaborn styles

    rng = np.random.RandomState(0)
    x = np.sort(10 * rng.rand(100))
    y = np.sin(2 * x) + 0.4 * rng.randn(100)

    plt.plot(x, y, 'o', color='gray', alpha=0.3)
    xfit = np.linspace(0, 10, 1000)

    for span in [0.05, 0.2, 0.5]:
        smoother = LinearSmoother(span)
        smoother.fit(x, y)
        plt.plot(xfit, smoother.predict(xfit),
                 label="LinearSmoother(span={0:.2f})".format(span))
    plt.legend()

It would be desirable to have some *automatic* way of choosing the best span
value for a particular dataset.
Further, we might imagine data for which this optimal span value could change
over the width of the plot!
Such adaptive auto-tuning is the goal of the Supersmoother algorithm, discussed
in the next section.
