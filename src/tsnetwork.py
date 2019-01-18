import random
import numpy as np


class Layer(object):

    activation = None

    def __init__(self, size, previous_layer=None):

        self.size = size

        if previous_layer:
            self.weights = np.random.randn(size, previous_layer.size)
            self.biases = np.random.randn(size, 1)

    def feedforward(self, input_activation):

        self.input_activation = input_activation
        self.z = np.dot(self.weights, input_activation) + self.biases
        self.activation = sigmoid(self.z)
        return self.activation

class Network(object):

    def __init__(self, layer_sizes):

        layer = Layer(layer_sizes[0]) # input layer
        self.layers = [layer]
        for layer_size in layer_sizes[1:]:
            layer = Layer(layer_size, layer)
            self.layers.append(layer)

    def SGD(self, X_train, y_train, epochs, mini_batch_size, eta, X_test=None, y_test=None):
        """
            Train the neural network using mini-batch stochastic gradient descent.
            The training_data is a list of tuples (x, y) representing the training inputs and
            the desired outputs.  The other non-optional parameters are
            self-explanatory.

            If test_data is provided then the network will be evaluated against the test data after
            each epoch, and partial progress printed out.  This is useful for
            tracking progress, but slows things down substantially.
        """

        training_data = list(zip(X_train, y_train))
        data_size = len(training_data)

        for epoch in range(epochs):

            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, data_size, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if X_test is not None and y_test is not None:
                print("Epoch {0}: {1} / {2}".format(epoch+1, self.evaluate(X_test, y_test), len(X_test)))
            else:
                print("Epoch {0} complete".format(epoch+1))

    def update_mini_batch(self, mini_batch, eta):
        """
            Update the network's weights and biases by applying
            gradient descent using backpropagation to a single mini batch.
            The mini_batch is a list of tuples (x, y), and eta is the learning rate.
        """

        # initialize weights, biases zero arrays
        nabla_b = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]   # starting from first hidden layer
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.layers[1:]]  # starting from first hidden layer

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        scaling_factor = eta / len(mini_batch)
        for layer, nw, nb in zip(self.layers[1:], nabla_w, nabla_b):
            # import pdb; pdb.set_trace()
            layer.weights -= scaling_factor * nw
            layer.biases -= scaling_factor * nb

    def backprop(self, x, y):
        """
            Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x.
            nabla_b and nabla_w are layer-by-layer lists of numpy arrays, same shape of self.biases and self.weights.
        """

        # only hidden layers have biases and weights
        nabla_cost_b = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]  # starting from first hidden layer
        nabla_cost_w = [np.zeros(layer.weights.shape) for layer in self.layers[1:]] # starting from first hidden layer

        # feedforward
        activation = x          # set the initial activation to be the input training data, x

        # for layer in self.layers[1:]:
        #     activation = layer.feedforward(activation)
        activation = self.feedforward(x)

        # backward process
        output_layer = self.layers[-1]
        delta = cost_derivative(output_layer.activation, y) * sigmoid_derivative(output_layer.z) # (σ(z) - y) * σ'(z)

        # import pdb; pdb.set_trace()
        nabla_cost_b[-1] = delta    # (σ(z) - y) * σ'(z)
        nabla_cost_w[-1] = np.dot(delta, self.layers[-2].activation.T)  # (σ(z) - y) * σ'(z) * previous_activation

        for layer_num in range(2, len(self.layers)):

            this_layer = self.layers[-layer_num]
            next_layer = self.layers[-layer_num+1]
                                # row: next layer, col: this layer -> row: this layer, col: next layer
            delta = np.dot(next_layer.weights.T, delta) * sigmoid_derivative(this_layer.z)
            nabla_cost_b[-layer_num] = delta
            nabla_cost_w[-layer_num] = np.dot(delta, this_layer.input_activation.T)

        return (nabla_cost_b, nabla_cost_w)

    def evaluate(self, X_test, y_test):
        """
            Return the number of test inputs for which the neural
            network outputs the correct result. Note that the neural
            network's output is assumed to be the index of whichever
            neuron in the final layer has the highest activation.
        """

        test_inputs = [np.reshape(x, (784, 1)) for x in X_test]
        test_results = [vectorized_result(y) for y in y_test]
        test_data = list(zip(test_inputs, test_results))
        test_results = [
            (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results)

    def feedforward(self, x):
        activation = x          # set the initial activation to be the input training data, x

        for layer in self.layers[1:]:
            activation = layer.feedforward(activation)

        return activation

def vectorized_result(j):
    """
        Return a 10-dimensional unit vector with a 1.0 in the jth
        position and zeroes elsewhere.  This is used to convert a digit
        (0...9) into a corresponding desired output from the neural
        network.
    """
    e = np.zeros((10, 1))
    e[j] = 1
    return e

def sigmoid(z):
    """The sigmoid function."""
    return (z/(1+np.abs(z)) + 1) * 0.5
    # return 1/(1+np.exp(-z))

def cost_derivative(output_activations, y):
    """
        Derivative of the sigmoid function with respect to output activation.
    """
    return (output_activations-y)

def sigmoid_derivative(z):
    """
        Derivative of the sigmoid function with respect to z.
    """
    return sigmoid(z)*(1-sigmoid(z))