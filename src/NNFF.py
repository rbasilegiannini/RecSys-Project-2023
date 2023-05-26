import numpy as np
import activation_functions as af
import random


class NeuralNetworkFF:
    def __init__(self,
                 input_dim,
                 neurons_per_hidden_layers=None,
                 neurons_per_output_layer=1,
                 activation_function='sigmoid',
                 bias=None):
        """
        :param input_dim:
            The input dimension.
        :param neurons_per_hidden_layers:
            For each hidden layer, the number of the hidden neurons (list of integers).
        :param neurons_per_output_layer:
            The number of the neurons in the output layer.
        :param activation_function:
            The activation function type. This parameter is expressed as a string (refer to activation functions module)
        :param bias:
            The bias flag.

        """
        if neurons_per_hidden_layers is None:
            neurons_per_hidden_layers = [1]

        self.__input_dim = input_dim
        self.__neurons_per_hidden_layers = neurons_per_hidden_layers
        self.__neurons_per_output_layer = neurons_per_output_layer
        self.__num_layers = len(neurons_per_hidden_layers) + 1  # sum also output layer
        self.__activation_function = activation_function
        self.__output_function = 'sigmoid'
        self.__bias = bias

        self.__neurons_per_layer = []
        self.__params_per_layer = []

        self.__init_weights()

    def __init_weights(self):
        """
        Weights random initialization.
        """

        # Init layers
        self.__neurons_per_layer = []
        for n in self.__neurons_per_hidden_layers:
            self.__neurons_per_layer.append(n)
        self.__neurons_per_layer.append(self.__neurons_per_output_layer)

        # For each layer we have a matrix of weights
        for layer in range(self.__num_layers):

            # Input layer
            if layer == 0:
                input_lines = self.__input_dim

            # Hidden or output layer
            else:
                input_lines = self.__neurons_per_layer[layer - 1]

            current_neurons = self.__neurons_per_layer[layer]
            weights = np.random.rand(current_neurons, input_lines) - 0.5

            if self.__bias:
                weights = np.insert(weights, 0, [random.random()], axis=1)

            self.__params_per_layer.append(weights)

    def compute_network(self, input_data):  # TODO: refactoring
        """
        This function computes the output of the NN based on the current parameters

        :param input_data:
            Array with input data.
        :return:
            The activation [index 0] and outputs [index 1] of all layers.
        """

        # Init input
        input_data = np.array(input_data)
        if self.__bias:
            input_data = np.insert(input_data, 0, [1], axis=0)
        input_data = np.column_stack([input_data])

        # From the input layer to the last hidden layer
        activation = []
        output = []
        input_lines = input_data
        afunc_type = self.__activation_function
        for layer in range(self.__num_layers - 1):

            # Init loop
            current_params = self.__params_per_layer[layer]
            num_of_neurons = self.__neurons_per_layer[layer]
            output_of_layer = np.zeros(num_of_neurons)

            # Compute activation and output
            activation_of_layer = current_params @ input_lines
            activation.append(activation_of_layer.flatten())

            for n in range(num_of_neurons):
                output_of_layer[n] = af.activation_function[afunc_type](activation_of_layer[n])
            output.append(np.array(output_of_layer))

            # Update input_lines for the next loop
            input_lines = output_of_layer
            if self.__bias:
                input_lines = np.insert(input_lines, 0, [1], axis=0)

        # Compute the output layer
        current_params = self.__params_per_layer[-1]
        num_of_neurons = self.__neurons_per_output_layer
        output_of_layer = np.zeros(num_of_neurons)

        activation_of_layer = current_params @ input_lines
        activation.append(activation_of_layer.flatten())

        out_func_type = self.__output_function
        for n in range(num_of_neurons):
            output_of_layer[n] = af.activation_function[out_func_type](activation_of_layer[n])
        output.append(np.array(output_of_layer))

        return [activation, output]

    def set_layer_weights(self, layer, new_layer_weights):
        new_weights_cols = new_layer_weights.shape[1]
        current_weights_cols = self.__params_per_layer[layer].shape[1]

        if self.__bias and (current_weights_cols == new_weights_cols + 1):
            self.__params_per_layer[layer][:, 1:] = new_layer_weights
        else:
            self.__params_per_layer[layer] = new_layer_weights

    def set_layer_bias(self, layer, new_layer_bias):
        if self.__bias:
            self.__params_per_layer[layer][:, 0] = new_layer_bias

    def set_layer_params(self, layer, new_layer_params):
        if self.__bias:
            new_layer_bias = new_layer_params[:, 0]
            new_layer_weights = new_layer_params[:, 1:]

            self.set_layer_bias(layer, new_layer_bias)
            self.set_layer_weights(layer, new_layer_weights)
        else:
            self.set_layer_weights(layer, new_layer_params)

    def set_all_params(self, new_params):
        self.__params_per_layer = new_params

    def get_num_layer(self):
        return self.__num_layers

    def get_activation_function(self):
        return self.__activation_function

    def get_output_function(self):
        return self.__output_function

    def get_bias_state(self):
        return self.__bias

    def get_all_params(self):
        return self.__params_per_layer

    def get_layer_params(self, layer):
        return self.__params_per_layer[layer]

    def get_layer_weights(self, layer):
        if self.__bias:
            return self.__params_per_layer[layer][:, 1:]
        else:
            return self.__params_per_layer[layer]

    def get_all_weights(self):
        all_weights = []
        for layer in range(self.__num_layers):
            all_weights.append(self.get_layer_weights(layer))
        return all_weights

    def get_layer_neurons(self, layer):
        return self.__neurons_per_layer[layer]

    def get_all_neurons(self):
        all_neurons = []
        for layer in range(self.__num_layers):
            all_neurons.append(self.get_layer_neurons(layer))
        return all_neurons
