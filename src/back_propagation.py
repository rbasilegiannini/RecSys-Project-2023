import numpy as np
import activation_functions as af
import error_functions as ef


class BackPropagation:
    """
    This class implements the back propagation technique.
    """

    def __init__(self, NN):

        self.__NN = NN
        self.__num_layers = NN.get_num_layer()
        self.__a_func_type = NN.get_activation_function()
        self.__output_func_type = NN.get_output_function()
        self.__all_weights = NN.get_all_weights()
        self.__all_neurons = NN.get_all_neurons()
        self.__num_of_output_neurons = self.__all_neurons[-1]

        self.__deltas_per_layer = np.ndarray(self.__num_layers, dtype=np.ndarray)

        num_of_params = 0
        for layer_weights in NN.get_all_params():
            num_of_params += layer_weights.size

        self._num_of_params = num_of_params
        self._bias_state = NN.get_bias_state()

    def compute_error_gradient(self, input_data, targets):
        """
         This method computes the error gradient with the back propagation technique.

         :param input_data:
            The NN's input (a numpy array).
         :param targets:
             The targets used to compute the error gradient (a numpy array).
         :return:
             The error gradient.
         """

        # Forward propagation step
        all_activation, all_output = self.__NN.compute_network(input_data)
        output_lines = all_output[-1]

        # Back propagation step
        self.__back_propagation(targets, all_activation, output_lines)

        # Compute the error gradient
        gradE = np.zeros(self._num_of_params)
        param = 0
        prev_output_layer = 0
        for layer in range(self.__num_layers):
            num_neurons_layer = self.__all_neurons[layer]
            delta_i = self.__deltas_per_layer[layer]

            if layer > 0:
                prev_output_layer = all_output[layer - 1]
            input_lines = self.__all_weights[layer].shape[1]

            for neuron in range(num_neurons_layer):

                if self._bias_state:
                    # Compute dE_dparm: biases
                    dE_dparm = delta_i[neuron] * 1
                    gradE[param] = dE_dparm
                    param += 1

                for conn in range(input_lines):
                    if layer == 0:
                        dE_dparm = delta_i[neuron] * input_data[conn]
                    else:
                        dE_dparm = delta_i[neuron] * prev_output_layer[conn]
                    gradE[param] = dE_dparm
                    param += 1

        return gradE

    def __back_propagation(self, targets, all_activation, output_lines):
        """
        This method is used to compute the deltas.

        :param output_lines:
            The NN's output lines.
        :param all_activation:
            The activation values for each layer.
        :param targets:
             The targets used to compute the error gradient (a numpy array).
        :return:
        """
        last_activation = all_activation[-1]

        # Compute delta output
        delta_out = np.zeros(self.__num_of_output_neurons)
        for line in range(self.__num_of_output_neurons):
            a_k = last_activation[line]
            y_k = output_lines[line]
            t_k = targets[line]

            delta_out_k = af.activation_function_der[self.__output_func_type](a_k) * ef.cross_entropy_derivative(y_k, t_k)
            delta_out[line] = delta_out_k
        self.__deltas_per_layer[-1] = delta_out

        # Compute internal deltas
        num_hidden_layer = self.__num_layers - 1
        for layer in reversed(range(num_hidden_layer)):
            num_neurons_layer = self.__all_neurons[layer]
            activation_layer = all_activation[layer]
            next_weights_layer = self.__all_weights[layer + 1]
            next_delta_row = self.__deltas_per_layer[layer + 1]

            delta_i = np.zeros(num_neurons_layer)
            for neuron in range(num_neurons_layer):

                a_i = activation_layer[neuron]
                next_weights_column = next_weights_layer[:, neuron]

                summation = next_delta_row @ next_weights_column
                delta_i_k = af.activation_function_der[self.__a_func_type](a_i) * summation
                delta_i[neuron] = delta_i_k

            self.__deltas_per_layer[layer] = delta_i

