from nn_layers import *
from nn_core import *
from nn_activate import *
from data_mnist import *

# Define the Neural Network Architecture
nn = NeuralNetwork(loss='cross_entropy')

nn.add(DenseLayer(input_size=28*28, output_size=128))
nn.add(ActivationLayer(relu, relu_derivative))
nn.add(DenseLayer(input_size=128, output_size=64))
nn.add(ActivationLayer(relu, relu_derivative))
nn.add(DenseLayer(input_size=64, output_size=10))
nn.add(ActivationLayer(softmax, lambda z: 1))

X_train, y_train, X_test, y_test = get_mnist()

nn.train(X_train, y_train, epochs=10, batch_size=16, learning_rate=0.01)
