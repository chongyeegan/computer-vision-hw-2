import numpy as np
from matplotlib import pyplot as plt
__author__ = 'jhh283'


# general plotting utility function
def plt_plot(x, y, xlabel, ylabel, title, filename):
    x, y = sort_x(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    C = plt.plot(x, y, ls='solid')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(filename, dpi=100)
    plt.show()
    plt.close()


# sorts np arrays according to increasing x
def sort_x(x, y):
    npa = np.array(zip(x, y))
    npa = npa[npa[:, 0].argsort()]
    return npa[:, 0], npa[:, 1]
