import numpy as np


def RPROP(NN, grad_E,
          update_max=50,
          update_min=1e-6,
          eta_minus=0.3,
          eta_plus=1.1):
    """
    This function implement the RPROP rule used to update the parameters of a NN based on the error gradient.

    :param NN:
        The NN to update.
    :param grad_E:
        The error gradient.
    :param eta_plus:
        The value of the update increment factor.
    :param eta_minus:
        The value of the update decrement factor.
    :param update_min:
        Min update value (used to avoid overflow/underflow problems)
    :param update_max:
        Max update value (used to avoid overflow/underflow problems)
    :return:
        The NN with updated parameters.
    """
    num_params = grad_E.size

    old_grad_E = np.zeros(num_params)
    update_value = np.full(num_params, 0.01)
    delta = np.zeros(num_params)
    num_layers = NN.get_num_layer()

    offset = 0
    for layer in range(num_layers):
        mat_params = NN.get_layer_params(layer)
        rows = mat_params.shape[0]
        cols = mat_params.shape[1]
        params = mat_params.flatten()

        for i in range(params.size):
            old_grad_E_i = old_grad_E[offset + i]
            current_grad_E_i = grad_E[offset + i]

            if (old_grad_E_i * current_grad_E_i) > 0:
                update_value[offset + i] = min(update_value[offset + i] * eta_plus, update_max)
                delta[offset + i] = -np.sign(current_grad_E_i) * update_value[offset + i]
                params[i] += delta[offset + i]
                old_grad_E_i = current_grad_E_i

            elif (old_grad_E_i * current_grad_E_i) < 0:
                params[i] -= delta[offset + i]  # Backtracking
                update_value[offset + i] = max(update_value[offset + i] * eta_minus, update_min)
                old_grad_E_i = 0

            else:
                delta[offset + i] = -np.sign(current_grad_E_i) * update_value[offset + i]
                params[i] += delta[offset + i]
                old_gradE_i = current_grad_E_i

        # Set new parameters
        mat_params = np.reshape(params, [rows, cols])
        NN.set_layer_params(layer, mat_params)

        offset += params.size

    return NN
