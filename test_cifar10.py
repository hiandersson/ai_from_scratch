from nn_layers import *
from nn_core import *
from nn_activate import *
from data_cifar10 import *

# Load and preprocess the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = load_cifar10()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the Neural Network Architecture
nn = NeuralNetwork(loss='cross_entropy')
nn.add(ConvLayer(input_shape=(3, 32, 32), num_filters=16, filter_size=3, stride=1, padding=1))
nn.add(ActivationLayer(relu, relu_derivative))
nn.add(MaxPoolingLayer(pool_size=2, stride=2))
nn.add(ConvLayer(input_shape=(16, 16, 16), num_filters=32, filter_size=3, stride=1, padding=1))
nn.add(ActivationLayer(relu, relu_derivative))
nn.add(MaxPoolingLayer(pool_size=2, stride=2))
nn.add(DenseLayer(32 * 8 * 8, 128))  # Flattened output from conv layers
nn.add(ActivationLayer(relu, relu_derivative))
nn.add(DenseLayer(128, 10))  # Output layer with 10 neurons for 10 classes
nn.add(ActivationLayer(softmax, lambda z: 1))  # Softmax activation in the output layer

# Train the Neural Network
nn.train(X_train, y_train, epochs=10, batch_size=32, learning_rate=0.01)