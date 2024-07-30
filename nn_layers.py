import numpy as np

class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class DenseLayer(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))
        self.verbose_forward = False
        self.verbose_backward = False

    def forward(self, input):

        self.stored_input = input

        # inputs is shaped (batch_size, input_size)

        # weights is shaped (input_size, output_size)

        # biases is shaped (1, output_size)

        # ---

        # > Compute: output from this layer to the next0 by multiplying inputs with weights
        # > Dot: product input (batch_size, input_size) x weights (input_size, output_size) = (batch_size, output_size)
        self.output = np.dot(input, self.weights) + self.biases

        if self.verbose_forward == True:
            print(f'forward input shape {self.stored_input.shape} weights shape {self.weights.shape} biases shape {self.biases.shape} output shape {self.output.shape}')

        return self.output

    def backward(self, output_gradient, learning_rate):

        # ---

        # > Compute: the gradient of the loss with respect to the input
        # > Dot: product output_gradient(batch_size, output_size) x self.weights transposed (output_size, input_size) = (batch_size, input_size)
        input_gradient = np.dot(output_gradient, self.weights.T) 

        if self.verbose_backward == True:
            print(f'backward output_gradient {output_gradient.shape} weights transposed {self.weights.T.shape}')

        # ---

        # > Compute: the gradient of the loss with respect to the weights
        # > Dot: product input transposed (input_size, batch_size) x output_gradient(batch_size, output_size) = (input_size, output_size)
        weights_gradient = np.dot(self.stored_input.T, output_gradient) 

        # > Compute: the gradient of the loss with respect to the biases
        # > Dot: sum over all examples in a batch
        biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        
        return input_gradient

class ActivationLayer(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_derivative(self.input)

class ConvLayer(Layer):
    def __init__(self, input_shape, num_filters, filter_size, stride=1, padding=0):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(num_filters, input_shape[0], filter_size, filter_size) * 0.1

    def forward(self, input):
        self.input = input
        self.input_padded = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        self.output_shape = (
            (self.input_padded.shape[2] - self.filter_size) // self.stride + 1,
            (self.input_padded.shape[3] - self.filter_size) // self.stride + 1
        )
        self.output = np.zeros((input.shape[0], self.num_filters, *self.output_shape))

        for i in range(0, self.output.shape[2], self.stride):
            for j in range(0, self.output.shape[3], self.stride):
                region = self.input_padded[:, :, i:i+self.filter_size, j:j+self.filter_size]
                self.output[:, :, i // self.stride, j // self.stride] = np.sum(region * self.filters, axis=(1, 2, 3))

        return self.output

    def backward(self, output_gradient, learning_rate):
        filter_gradient = np.zeros(self.filters.shape)
        input_gradient_padded = np.zeros(self.input_padded.shape)

        for i in range(0, self.output.shape[2], self.stride):
            for j in range(0, self.output.shape[3], self.stride):
                region = self.input_padded[:, :, i:i+self.filter_size, j:j+self.filter_size]
                for k in range(self.num_filters):
                    filter_gradient[k] += np.sum(region * (output_gradient[:, k, i // self.stride, j // self.stride])[:, None, None, None], axis=0)
                for n in range(output_gradient.shape[0]):
                    input_gradient_padded[n, :, i:i+self.filter_size, j:j+self.filter_size] += np.sum((self.filters * (output_gradient[n, :, i // self.stride, j // self.stride])[:, None, None, None]), axis=0)

        self.filters -= learning_rate * filter_gradient
        if self.padding != 0:
            input_gradient = input_gradient_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            input_gradient = input_gradient_padded

        return input_gradient

class MaxPoolingLayer(Layer):
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        self.output_shape = (
            (input.shape[2] - self.pool_size) // self.stride + 1,
            (input.shape[3] - self.pool_size) // self.stride + 1
        )
        self.output = np.zeros((input.shape[0], input.shape[1], *self.output_shape))

        for i in range(0, self.output.shape[2], self.stride):
            for j in range(0, self.output.shape[3], self.stride):
                region = input[:, :, i:i+self.pool_size, j:j+self.pool_size]
                self.output[:, :, i // self.stride, j // self.stride] = np.max(region, axis=(2, 3))

        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros(self.input.shape)

        for i in range(0, self.output.shape[2], self.stride):
            for j in range(0, self.output.shape[3], self.stride):
                region = self.input[:, :, i:i+self.pool_size, j:j+self.pool_size]
                max_region = np.max(region, axis=(2, 3), keepdims=True)
                region_mask = (region == max_region)
                input_gradient[:, :, i:i+self.pool_size, j:j+self.pool_size] += region_mask * (output_gradient[:, :, i // self.stride, j // self.stride])[:, :, None, None]

        return input_gradient