import matplotlib.pyplot as plt
import generator as gen
import pickle
import tensorflow as tf
import numpy as np

def view():
    obj3d = gen.generate_cylinder()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(obj3d[:, 0], obj3d[:, 1], obj3d[:, 2])
    plt.show()


if __name__ == '__main__':
    a = tf.convert_to_tensor([1, 0, 0, 0])
    b = tf.convert_to_tensor([0.2, 0.4, 0.3, 0.1])
    cc = tf.keras.losses.CategoricalCrossentropy()
    print(np.eye(5)[2])