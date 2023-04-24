import numpy as np
import matplotlib.pyplot as plt
import back_propagation as bp
import error_functions as ef


class NetConfigEvaluated:
    """
    This data structure contains information about an evaluated NN.
    """
    net_params = []
    training_error = 0
    validation_error = 0

    def __init__(self, net_params, training_error, validation_error):
        """
        :param net_params:
            All parameters of the evaluated NN. This input is a list of matrix (one matrix per layer).
        :param training_error:
            The training error of the NN.
        :param validation_error:
            The validation error of the NN.
        """
        self.net_params = net_params
        self.training_error = training_error
        self.validation_error = validation_error


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


def compute_error(NN, samples, labels_one_hot):
    """
    This function computes the cross entropy soft max error on a specific dataset.

    :param NN:
        The NN to be evaluated.
    :param samples:
        The dataset's samples. This input must be a matrix where each row is a sample and each column is a feature.
    :param labels_one_hot:
        The samples' labels. Each row is the label (in one-hot encoding) of the related sample (same row in "samples")
    :return:
        The error's value.
    """

    dataset = {
        'samples': samples[:, :],
        'label': labels_one_hot[:, :]
    }

    error = 0
    dataset_size = samples.shape[0]
    for i in range(dataset_size):
        all_output = NN.compute_network(dataset['samples'][i])[1]
        output_net = all_output[-1]
        target = np.column_stack([dataset['label'][i]])
        sample_error = ef.cross_entropy_loss(ef.softmax(output_net), target)
        error += sample_error

    return error/dataset_size


def learning(NN, max_epoch, train_samples, labels_one_hot):
    """
    This function is used to train a NN with a specific training set.

    :param NN:
        The NN that is about to be trained.
    :param max_epoch:
        The max number of the epochs.
    :param train_samples:
        The training set's samples. This input must be a matrix where each row is a sample and each column is a feature.
    :param labels_one_hot:
        The samples' labels. Each row is the label (in one-hot encoding) of the related sample (same row in "samples")
    :return:
        The trained NN.
    """

    train_samples = np.array(train_samples)
    labels_one_hot = np.array(labels_one_hot)
    num_samples = train_samples.shape[0]
    training_set_size = round(num_samples * 0.7)

    training_set = {
        'samples': train_samples[:training_set_size, :],
        'label': labels_one_hot[:training_set_size, :]
    }
    validation_set = {
        'samples': train_samples[training_set_size + 1:, :],
        'label': labels_one_hot[training_set_size+1:, :]
    }

    # Compute the number of the NN parameters
    num_params = 0
    for layer in range(NN.get_num_layer()):
        num_params += NN.get_layer_params(layer).size

    # Learning
    net_params_list = []
    for epoch in range(max_epoch):

        # Compute the error gradient
        grad_E_tot = np.zeros(num_params)
        for i in range(training_set_size):
            grad_E_sample = bp.back_propagation(NN, training_set['samples'][i], training_set['label'][i])
            grad_E_tot += grad_E_sample

        # Update NN parameters
        NN = RPROP(NN, grad_E_tot)

        # Compute errors
        train_error = compute_error(NN, training_set['samples'], training_set['label'])
        val_error = compute_error(NN, validation_set['samples'], validation_set['label'])

        # Save the evaluated NN's information of this epoch
        net_params = NetConfigEvaluated(NN.get_all_params(), train_error, val_error)
        net_params_list.append(net_params)

    # Retrieve best epoch (smallest validation error)
    all_validation_errors = [net.validation_error for net in net_params_list]
    all_validation_errors = np.array(all_validation_errors)
    best_epoch = np.argmin(all_validation_errors)

    # Update NN with best parameters
    NN.set_all_params(net_params_list[best_epoch].net_params)

    # DEBUG
    plot_errors(net_params_list)

    return NN


def plot_errors(net_config_evaluated_list):
    """
    This function is used to plot training and validation errors of an evaluated NN.

    :param net_config_evaluated_list:
        The list of the NN's configuration (NetConfigEvaluated class), one element per epoch.
    """

    x = []
    e_train = []
    e_val = []
    for i in range(len(net_config_evaluated_list)):
        x.append(i)
        e_train.append(net_config_evaluated_list[i].training_error)
        e_val.append(net_config_evaluated_list[i].validation_error)

    plt.plot(x, e_train, label='train')
    plt.plot(x, e_val, label='val')
    plt.legend()
    plt.ylabel('Error')

    plt.show()


def accuracy(NN, samples, labels_one_hot):
    """
    This function computes the accuracy of a NN on a specific dataset.

    :param NN:
        The NN to be evaluated.
    :param samples:
        The dataset's samples. This input must be a matrix where each row is a sample and each column is a feature.
    :param labels_one_hot:
        The samples' labels. Each row is the label (in one-hot encoding) of the related sample (same row in "samples")

    :return:
        The accuracy as a percentage.
    """

    num_correct = 0
    num_samples = samples.shape[0]
    for i in range(num_samples):
        output = (NN.compute_network(samples[i])[1])[-1]

        # Retrieve the most probable class
        output = ef.softmax(output)
        predict_class = np.argmax(output)

        # Compare
        target = labels_one_hot[i]
        if target[predict_class] == 1:
            num_correct += 1

    acc = (num_correct / num_samples) * 100

    return acc


def normalize_samples(samples, l_ext, r_ext):
    """
    This function is used to normalize each feature of the samples in [l_ext, r_ext]

    :param samples:
        The dataset's samples. This input must be a matrix where each row is a sample and each column is a feature.
    :param l_ext:
        left end of the interval.
    :param r_ext:
        Right end of the interval.
    :return:
        The normalized samples.
    """

    num_samples = samples.shape[0]
    max_value = samples.max()
    min_value = samples.min()

    for i in range(num_samples):
        samples[i] = l_ext + (((samples[i] - min_value) * (r_ext - l_ext)) / (max_value - min_value))

    return samples