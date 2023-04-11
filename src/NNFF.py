import numpy as np


class NeuralNetworkFF:
    def __init__(self,
                 input_dim,
                 neurons_per_input_layer=1,
                 neurons_per_hidden_layers=None,
                 neurons_per_output_layer=1,
                 bias=None):
        """
        :param input_dim:
            The input dimension.
        :param neurons_per_input_layer:
            The number of the neurons in the input layer.
        :param neurons_per_hidden_layers:
            For each hidden layer, the number of the hidden neurons.
        :param neurons_per_output_layer:
            The number of the neurons in the output layer.
        """
        if neurons_per_hidden_layers is None:
            neurons_per_hidden_layers = [1]

        self.__input_dim = input_dim
        self.__neurons_per_input_layer = neurons_per_input_layer
        self.__neurons_per_hidden_layers = neurons_per_hidden_layers
        self.__neurons_per_output_layer = neurons_per_output_layer
        self.__num_layers = len(neurons_per_hidden_layers) + 2  # sum input and output layers
        self.__bias = bias
        self.__init_weights()

    def __init_weights(self):
        """
        Weights random initialization.
        """

        # For each layer we have a matrix of weights
        self.__weights_per_layer = []

        # Init layers
        neurons_per_layer = [self.__neurons_per_input_layer]
        for n in self.__neurons_per_hidden_layers:
            neurons_per_layer.append(n)
        neurons_per_layer.append(self.__neurons_per_output_layer)

        for layer in range(self.__num_layers):

            # Input layer
            if layer == 0:
                input_lines = self.__input_dim

            # Hidden or output layer
            else:
                input_lines = neurons_per_layer[layer - 1]

            current_neurons = neurons_per_layer[layer]
            weights = np.random.rand(current_neurons, input_lines)

            # Eventually add bias nodes
            if self.__bias:
                bias_nodes = np.ones((current_neurons, 1))
                weights = np.append(weights, bias_nodes, axis=1)

            self.__weights_per_layer.append(weights)

