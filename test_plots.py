import numpy as np
import matplotlib.pyplot as plt
from supersmoother import MovingAverageSmoother, FixedSpanSmoother


def make_test_set(N=200, err_factor=0.1, rseed=None):
    rng = np.random.RandomState(rseed)
    x = 6 * rng.rand(N)
    dy = err_factor * np.sqrt(x)
    y = np.sin(x ** 2) + dy * rng.rand(N)
    return x, y, dy


def plot_simple():
    x, y, dy = make_test_set(rseed=0)

    xfit = np.linspace(0, 6, 1000)
    model = MovingAverageSmoother(0.1).fit(x, y, dy)
    yfit = model.predict(xfit)
    yfit2 = model.predict_slow(xfit)

    print(np.allclose(yfit, yfit2))

    plt.errorbar(x, y, dy, fmt='.', alpha=0.3)
    plt.plot(xfit, yfit, '-k')
    plt.plot(xfit, yfit2, '-k')


def plot_linear():
    x, y, dy = make_test_set(rseed=0)

    xfit = np.linspace(0, 6, 1000)
    model = FixedSpanSmoother(0.05).fit(x, y, dy)
    yfit = model.predict(xfit)
    yfit2 = model.predict_slow(xfit)

    print(np.allclose(yfit, yfit2))

    plt.errorbar(x, y, dy, fmt='.', alpha=0.3)
    plt.plot(xfit, yfit, '-k')
    plt.plot(xfit, yfit2, '-k')


plt.figure()
plot_simple()

plt.figure()
plot_linear()
plt.show()
