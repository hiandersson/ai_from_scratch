import numpy as np
import urllib.request
import tarfile
import os
import pickle

# Function to download and extract CIFAR-10 dataset
def download_and_extract_cifar10():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    folder = "cifar-10-batches-py"

    if not os.path.exists(folder):
        urllib.request.urlretrieve(url, filename)
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()

# Function to load CIFAR-10 data
def load_cifar10_batch(filename):
    with open(filename, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
        data = batch['data'].reshape((10000, 3, 32, 32)).astype('float32') / 255.0
        labels = np.array(batch['labels'])
        return data, labels

# Function to load all CIFAR-10 data
def load_cifar10():
    download_and_extract_cifar10()
    X_train, y_train = [], []
    for i in range(1, 6):
        data, labels = load_cifar10_batch(f'cifar-10-batches-py/data_batch_{i}')
        X_train.append(data)
        y_train.append(labels)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test, y_test = load_cifar10_batch('cifar-10-batches-py/test_batch')
    return (X_train, y_train), (X_test, y_test)

# One-hot encoding function
def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

# Load and preprocess the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = load_cifar10()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)