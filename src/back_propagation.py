import numpy as np

import NNFF as nnff
import activation_functions as af
import error_functions as ef


def back_propagation(NN, input_data, targets):
    NN = nnff.NeuralNetworkFF(NN)

    num_layers = NN.get_num_layer()
    a_func_type = NN.get_activation_function()
    output_func_type = NN.get_output_function()
    all_weights = NN.get_all_weights()
    all_neurons = NN.get_all_neurons()
    [all_activation, all_output] = NN.compute_network(input_data)
    output_lines = all_output[-1]
    num_of_output_neurons = len(all_neurons[-1])
    last_activation = all_activation[-1]

    all_delta = []

    # Post-processing step
    output_lines = ef.softmax(output_lines)

    # Compute delta output
    delta_out = []
    for line in range(num_of_output_neurons):
        a_k = last_activation[line]
        y_k = output_lines[line]
        t_k = targets[line]

        delta_out_k = af.activation_function_der[output_func_type](a_k) * ef.cross_entropy_soft_max_der(y_k, t_k)
        delta_out.append(delta_out_k)
    all_delta.append(np.array([delta_out]))

    # Compute internal delta
    num_hidden_layer = num_layers - 1
    for layer in reversed(range(num_hidden_layer)):
        num_neurons_layer = all_neurons[layer]
        activation_layer = all_activation[layer]
        next_weights_layer = all_weights[layer+1]

        delta_i = []
        for neuron in range(num_neurons_layer):
            a_i = activation_layer[neuron]
            next_delta_row = all_delta[layer+1]
            next_weights_column = next_weights_layer[:, neuron]

            summation = next_delta_row @ next_weights_column
            delta_i_k = af.activation_function_der[a_func_type](a_i) * summation
            delta_i.append(delta_i_k)

        all_delta.insert(0, np.array([delta_i]))

    # Compute error gradient
