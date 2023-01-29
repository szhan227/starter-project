import tensorflow as tf
import numpy as np
import pickle
from model import PointNet
import os
import glob
import trimesh


def run_modelnet10():
    DATA_DIR = tf.keras.utils.get_file(
        "modelnet.zip",
        "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        extract=True,
    )
    DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")

    def parse_dataset(num_points=2048):

        train_points = []
        train_labels = []
        test_points = []
        test_labels = []
        class_map = {}
        folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

        for i, folder in enumerate(folders):
            print("processing class: {}".format(os.path.basename(folder)))
            # store folder name with ID so we can retrieve later
            class_map[i] = folder.split("/")[-1]
            # gather all files
            train_files = glob.glob(os.path.join(folder, "train/*"))
            test_files = glob.glob(os.path.join(folder, "test/*"))

            for f in train_files:
                train_points.append(trimesh.load(f).sample(num_points))
                train_labels.append(i)

            for f in test_files:
                test_points.append(trimesh.load(f).sample(num_points))
                test_labels.append(i)

        return (
            np.array(train_points),
            np.array(test_points),
            np.array(train_labels),
            np.array(test_labels),
            class_map,
        )

    NUM_POINTS = 2048
    NUM_CLASSES = 10
    BATCH_SIZE = 32

    train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
        NUM_POINTS
    )

    def augment(points, label):
        # jitter points
        points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
        # shuffle points
        points = tf.random.shuffle(points)
        return points, label

    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

    train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
    test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

    model = PointNet(num_class=10)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_dataset, epochs=1, batch_size=50, validation_data=test_dataset)

def run_mine():
    print('start of run_mine')
    with open('data3dshape_2048_500.p', 'rb') as f:
        data = pickle.load(f)

    BATCH_SIZE = 20

    # inputs = tf.convert_to_tensor(data['inputs'], dtype=tf.float32)
    # labels = tf.convert_to_tensor(data['labels'], dtype=tf.float32)
    inputs = data['inputs']
    labels = data['labels']
    print(inputs.shape)
    train_inputs = inputs[:-100]
    train_labels = labels[:-100]
    test_inputs = inputs[-100:]
    test_labels = labels[-100:]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))

    train_dataset = train_dataset.shuffle(len(train_inputs)).batch(BATCH_SIZE)
    test_dataset = test_dataset.shuffle(len(test_inputs)).batch(BATCH_SIZE)

    model = PointNet(num_class=5)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.fit(train_dataset, epochs=10, batch_size=BATCH_SIZE, validation_data=test_dataset)
    print('end of run_mine')


def test_code():
    model = PointNet(7)
    with open('data_14.p', 'rb') as f:
        data = pickle.load(f)

    inputs = data['inputs']
    labels = data['labels']
    print(inputs.shape)
    # ts = tf.convert_to_tensor(inputs, dtype=tf.float32)
    ts = inputs
    print(ts.shape)
    # ts = tf.expand_dims(ts, axis=0)
    print(ts.shape)

    output = model.call(ts)
    print(output.shape)


if __name__ == '__main__':

    run_mine()
    # test_code()
