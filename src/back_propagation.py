import numpy as np

import NNFF as nnff
import activation_functions as af
import error_functions as ef


def back_propagation(NN, input_data, targets):
    targets = np.column_stack([targets])    # To uniform

    num_layers = NN.get_num_layer()
    a_func_type = NN.get_activation_function()
    output_func_type = NN.get_output_function()
    all_weights = NN.get_all_weights()
    all_neurons = NN.get_all_neurons()
    [all_activation, all_output] = NN.compute_network(input_data)
    output_lines = all_output[-1]
    num_of_output_neurons = all_neurons[-1]
    last_activation = all_activation[-1]

    all_delta = []

    # Post-processing step
    output_lines = ef.softmax(output_lines)

    # Compute delta output
    delta_out = []
    for line in range(num_of_output_neurons):
        a_k = last_activation[line][0]
        y_k = output_lines[line][0]
        t_k = targets[line][0]

        delta_out_k = af.activation_function_der[output_func_type](a_k) * ef.cross_entropy_soft_max_der(y_k, t_k)
        delta_out.append(delta_out_k)
    all_delta.append(np.array(delta_out))

    # Compute internal delta
    num_hidden_layer = num_layers - 1
    for layer in reversed(range(num_hidden_layer)):
        num_neurons_layer = all_neurons[layer]
        activation_layer = all_activation[layer]
        next_weights_layer = all_weights[layer+1]
        next_delta_row = all_delta[0]   # Take the last element insert in front

        delta_i = []
        for neuron in range(num_neurons_layer):
            a_i = activation_layer[neuron][0]
            next_weights_column = next_weights_layer[:, neuron]

            summation = next_delta_row @ next_weights_column
            delta_i_k = af.activation_function_der[a_func_type](a_i) * summation
            delta_i.append(delta_i_k)

        all_delta.insert(0, np.array(delta_i))  # Front

    # Compute error gradient
    gradE = []
    output_layer = 0
    for layer in range(num_layers):
        num_neurons_layer = all_neurons[layer]
        delta_i = all_delta[layer]
        weights_layer = all_weights[layer]
        if layer > 0:
            output_layer = all_output[layer-1]

        input_lines = weights_layer.shape[1]

        for neuron in range(num_neurons_layer):

            if NN.get_bias_state():
                # Compute dE_dparm: biases
                dE_dparm = delta_i[neuron] * 1
                gradE.append(dE_dparm)

            for conn in range(input_lines):
                if layer == 0:
                    dE_dparm = delta_i[neuron] * input_data[conn]
                else:
                    dE_dparm = delta_i[neuron] * output_layer[conn][0]
                gradE.append(dE_dparm)

    return np.array(gradE)


