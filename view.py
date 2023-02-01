import matplotlib.pyplot as plt
import generator as gen
import pickle
import tensorflow as tf
import numpy as np
import scipy
from scipy import signal

def view():
    obj3d = gen.generate_cylinder()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(obj3d[:, 0], obj3d[:, 1], obj3d[:, 2])
    plt.show()


if __name__ == '__main__':
    pass

