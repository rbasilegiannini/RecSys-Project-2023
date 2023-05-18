import numpy as np

import activation_functions as af
import error_functions as ef


def back_propagation(NN, input_data, targets):
    """
    This function computes the error gradient with the back propagation technique.
    :param NN:
        The NN involved in the computation.
    :param input_data:
        The NN's input (a numpy array).
    :param targets:
        The targets used to compute the error gradient (a numpy array).
    :return:
        The error gradient.
    """

    num_layers = NN.get_num_layer()
    a_func_type = NN.get_activation_function()
    output_func_type = NN.get_output_function()
    all_weights = NN.get_all_weights()
    all_neurons = NN.get_all_neurons()
    [all_activation, all_output] = NN.compute_network(input_data)
    num_of_output_neurons = all_neurons[-1]

    # Convert in columns
   # for layer in range(num_layers):
   #     all_activation[layer] = np.column_stack([all_activation[layer]])
   #     all_output[layer] = np.column_stack([all_output[layer]])
   # targets = np.column_stack([targets])
   # input_data = np.column_stack([input_data])

    output_lines = all_output[-1]
    last_activation = all_activation[-1]

    # deltas_per_layer = []
    deltas_per_layer = np.ndarray(num_layers, dtype=np.ndarray)

    # Compute delta output
    # delta_out = []
    delta_out = np.zeros(num_of_output_neurons)
    for line in range(num_of_output_neurons):
        #a_k = last_activation[line][0]
        #y_k = output_lines[line][0]
        #t_k = targets[line][0]

        a_k = last_activation[line]
        y_k = output_lines[line]
        t_k = targets[line]

        delta_out_k = af.activation_function_der[output_func_type](a_k) * ef.cross_entropy_derivative(y_k, t_k)
        #delta_out.append(delta_out_k)
        delta_out[line] = delta_out_k
    #deltas_per_layer.append(delta_out)
    deltas_per_layer[-1] = delta_out

    # Compute internal delta
    num_hidden_layer = num_layers - 1
    for layer in reversed(range(num_hidden_layer)):
        num_neurons_layer = all_neurons[layer]
        activation_layer = all_activation[layer]
        next_weights_layer = all_weights[layer+1]
        #next_delta_row = deltas_per_layer[0]   # Take the last element insert in front
        next_delta_row = deltas_per_layer[layer+1]

        #delta_i = []
        delta_i = np.zeros(num_neurons_layer)
        for neuron in range(num_neurons_layer):
            # a_i = activation_layer[neuron][0]
            a_i = activation_layer[neuron]
            next_weights_column = next_weights_layer[:, neuron]

            summation = next_delta_row @ next_weights_column
            delta_i_k = af.activation_function_der[a_func_type](a_i) * summation
            #delta_i.append(delta_i_k)
            delta_i[neuron] = delta_i_k

        #all_delta.insert(0, np.array(delta_i))  # Front
        #deltas_per_layer.insert(0, delta_i)
        deltas_per_layer[layer] = delta_i

    # Compute error gradient
    num_of_params = 0
    for layer_weights in NN.get_all_params():
        num_of_params += layer_weights.size
    gradE = np.zeros(num_of_params)
    param = 0

    # gradE = []
    prev_output_layer = 0
    for layer in range(num_layers):
        num_neurons_layer = all_neurons[layer]
        delta_i = deltas_per_layer[layer]

        if layer > 0:
            prev_output_layer = all_output[layer-1]
        input_lines = all_weights[layer].shape[1]

        for neuron in range(num_neurons_layer):

            if NN.get_bias_state():
                # Compute dE_dparm: biases
                dE_dparm = delta_i[neuron] * 1
                #gradE.append(dE_dparm)
                gradE[param] = dE_dparm
                param += 1

            for conn in range(input_lines):
                if layer == 0:
                    #dE_dparm = delta_i[neuron] * input_data[conn][0]
                    dE_dparm = delta_i[neuron] * input_data[conn]
                else:
                    # dE_dparm = delta_i[neuron] * prev_output_layer[conn][0]
                    dE_dparm = delta_i[neuron] * prev_output_layer[conn]
                #gradE.append(dE_dparm)
                gradE[param] = dE_dparm
                param += 1

    return gradE


