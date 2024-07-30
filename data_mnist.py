import numpy as np
import os

def get_mnist():

    # Paths to the manually downloaded files
    train_images_path = './mnist_data/train-images.idx3-ubyte'
    train_labels_path = './mnist_data/train-labels.idx1-ubyte'
    test_images_path = './mnist_data/t10k-images.idx3-ubyte'
    test_labels_path = './mnist_data/t10k-labels.idx1-ubyte'

    # Function to load MNIST images
    def load_images(filename):
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            return data

    # Function to load MNIST labels
    def load_labels(filename):
        with open(filename, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
            return labels

    # Load the datasets
    X_train = load_images(train_images_path)
    y_train = load_labels(train_labels_path)
    X_test = load_images(test_images_path)
    y_test = load_labels(test_labels_path)

    # Flatten the images
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # One-hot encode the labels
    def to_categorical(y, num_classes):
        return np.eye(num_classes)[y]

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return X_train, y_train, X_test, y_test