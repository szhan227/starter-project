"""
Created on 2023-01-23
By Siyang Zhang
Edited on 2023-02-01
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pickle

obj2idx = {'ball': 0, 'cube': 1, 'cylinder': 2, 'cone': 3, 'pyramid': 4, 'ring': 5, 'torus': 6}
idx2obj = {0: 'ball', 1: 'cube', 2: 'cylinder', 3: 'cone', 4: 'pyramid', 5: 'ring', 6: 'torus'}

matplotlib.use('TkAgg')


NUM_SAMPLES = 2048


def generate_dataset(num=1000, save=False):
    inputs = []
    labels = []

    for i, _ in enumerate(range(num)):
        inputs.append(rotate3D(generate_ball()))
        inputs.append(rotate3D(generate_cube()))
        inputs.append(rotate3D(generate_cylinder()))
        inputs.append(rotate3D(generate_cone()))
        inputs.append(rotate3D(generate_pyramid()))
        inputs.append(rotate3D(generate_ring()))
        inputs.append(rotate3D(generate_torus()))
        labels.extend([0, 1, 2, 3, 4, 5, 6])
        print('\rGenerating data: {}/{}'.format(i + 1, num), end='')
    print()

    inputs = np.array(inputs)
    labels = np.array(labels)
    print('inputs.shape: ', inputs.shape)
    print('labels.shape: ', labels.shape)
    inputs, labels = shuffle(inputs, labels)

    if save:
        data = {'inputs': inputs, 'labels': labels}
        with open(f'data3dshape_{NUM_SAMPLES}_{num * len(lb)}.p', 'wb') as f:
            pickle.dump(data, f)
    return inputs, labels


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate3D(obj3d):

    x = obj3d[:, 0]
    y = obj3d[:, 1]
    z = obj3d[:, 2]

    axis = np.random.randint(5, size=3)
    theta = np.random.uniform(0, 2 * np.pi)
    x1, y1, z1 = np.dot(rotation_matrix(axis, theta), np.stack((x, y, z), axis=0))
    return np.stack((x1, y1, z1), axis=1)

def generate_ball():
    """
    Generate a ball of data in 3D space
    """

    result = []
    R = np.random.uniform(0.5, 1.0)
    for _ in range(NUM_SAMPLES):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.random.uniform(0, R)
        # r = R
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        result.append([x, y, z])

    return np.array(result)


def generate_cube():

    result = []
    for _ in range(NUM_SAMPLES):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)
        result.append([x, y, z])

    return np.array(result)


def generate_cylinder():

    result = []
    r = np.random.uniform(0.1, 0.5)
    h = np.random.uniform(0.1, 5.0)

    for _ in range(NUM_SAMPLES):
        theta = np.random.uniform(0, 2 * np.pi)
        a = np.random.uniform(0, r)
        z = np.random.uniform(0, h)
        x = a * np.cos(theta)
        y = a * np.sin(theta)
        result.append([x, y, z])

    return np.array(result)


def generate_cone():
    result = []
    r = np.random.uniform(0.1, 5.0)
    h = np.random.uniform(0.1, 5.0)

    for _ in range(NUM_SAMPLES):
        z = np.random.uniform(0, h)
        rh = (h - z) * r / h
        a = np.random.uniform(0, rh)
        # a = rh
        theta = np.random.uniform(0, 2 * np.pi)
        x = a * np.cos(theta)
        y = a * np.sin(theta)
        result.append([x, y, z])

    return np.array(result)


def generate_pyramid():
    result = []
    a = np.random.uniform(0.1, 5.0)
    h = np.random.uniform(0.1, 5.0)

    for _ in range(NUM_SAMPLES):
        x = np.random.uniform(-a, a)
        y = np.random.uniform(-a, a)
        z = np.random.uniform(0, h)

        xh = (h - z) * x / h
        yh = (h - z) * y / h

        result.append([xh, yh, z])

    return np.array(result)


def generate_ring():
    result = []
    r = np.random.uniform(2.0, 3.0)
    R = np.random.uniform(4.0, 5.0)

    h = np.random.uniform(0.1, 1.0)

    for _ in range(NUM_SAMPLES):

        theta = np.random.uniform(0, 2 * np.pi)
        a = np.random.uniform(r, R)
        x = a * np.cos(theta)
        y = a * np.sin(theta)
        z = np.random.uniform(-h, h)

        result.append([x, y, z])

    return np.array(result)


def generate_torus():
    result = []
    r = np.random.uniform(1.0, 3.0)
    R = np.random.uniform(4.0, 5.0)

    for _ in range(NUM_SAMPLES):

        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        a = np.random.uniform(0, r)
        # a = r
        x = (R + a * np.cos(phi)) * np.cos(theta)
        y = (R + a * np.cos(phi)) * np.sin(theta)
        z = a * np.sin(phi)

        result.append([x, y, z])

    return np.array(result)


if __name__ == "__main__":
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # obj3d = generate_torus()

    # print(obj3d.shape)
    # ax.scatter3D(obj3d[:, 0], obj3d[:, 1], obj3d[:, 2], c=obj3d[:, 2], cmap='Greens')


    # plt.show()
    inputs, labels = generate_dataset(100, save=True)
    # with open('data_14.p', 'rb') as f:
    #     data = pickle.load(f)
    #     inputs = data['inputs']
    #     labels = data['labels']
    #     print(inputs, labels)



