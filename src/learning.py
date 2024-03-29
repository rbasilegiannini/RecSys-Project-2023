import numpy as np
import matplotlib.pyplot as plt
import back_propagation as bp
import error_functions as ef
import time


class EvaluatedNetConfig:
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


class RPROP:
    """
    This class implements the RPROP rule used to update the parameters of a NN based on the error gradient.
    """

    def __init__(self, num_params):
        self.__old_grad_E = np.zeros(num_params)
        self.__update_max = 50
        self.__update_min = 1e-6
        self.__eta_minus = 0.3
        self.__eta_plus = 1.1
        self.__update_value = np.full(num_params, 0.01)
        self.__delta = np.zeros(num_params)

    def update(self, NN, grad_E):
        """
        This method is used to update parameters of the NN.

        :param NN:
            The NN to update.
        :param grad_E:
            The current error gradient.

        :return:
            The NN with updated parameters.
        """

        num_layers = NN.get_num_layer()
        offset = 0
        for layer in range(num_layers):
            mat_params = NN.get_layer_params(layer)
            rows = mat_params.shape[0]
            cols = mat_params.shape[1]
            params = mat_params.flatten()

            for i in range(params.size):

                old_grad_E_i = self.__old_grad_E[offset + i]
                current_grad_E_i = grad_E[offset + i]

                if (old_grad_E_i * current_grad_E_i) > 0:
                    self.__update_value[offset + i] = min(self.__update_value[offset + i] * self.__eta_plus,
                                                          self.__update_max)
                    self.__delta[offset + i] = -np.sign(current_grad_E_i) * self.__update_value[offset + i]
                    params[i] += self.__delta[offset + i]
                    old_grad_E_i = current_grad_E_i

                elif (old_grad_E_i * current_grad_E_i) < 0:
                    params[i] -= self.__delta[offset + i]  # Backtracking
                    self.__update_value[offset + i] = max(self.__update_value[offset + i] * self.__eta_minus,
                                                          self.__update_min)
                    old_grad_E_i = 0

                else:
                    self.__delta[offset + i] = -np.sign(current_grad_E_i) * self.__update_value[offset + i]
                    params[i] += self.__delta[offset + i]
                    old_grad_E_i = current_grad_E_i

                self.__old_grad_E[offset + i] = old_grad_E_i

            # Set new parameters
            mat_params = np.reshape(params, [rows, cols])
            NN.set_layer_params(layer, mat_params)

            offset += params.size

        return NN


def _compute_error(NN, samples, scalar_labels):
    """
    This function computes the cross entropy soft max error on a specific dataset.

    :param NN:
        The NN to be evaluated.
    :param samples:
        The dataset's samples. This input must be a matrix where each row is a sample and each column is a feature.
    :param scalar_labels:
        The samples' labels. Each row is a scalar label of the related sample (same row in "samples")
    :return:
        The error's value.
    """

    error = 0
    dataset_size = samples.shape[0]
    for i in range(dataset_size):
        all_output = NN.compute_network(samples[i])[1]
        output_net = all_output[-1][0]
        target = scalar_labels[i]
        sample_error = ef.cross_entropy_loss_per_sample(output_net, target)
        error += sample_error

    return error / dataset_size


def learning(NN, max_epoch, train_samples, scalar_labels):
    """
    This function is used to train a NN with a specific training set.

    :param NN:
        The NN that is about to be trained.
    :param max_epoch:
        The max number of the epochs.
    :param train_samples:
        The training set's samples. This input must be a matrix where each row is a sample and each column is a feature.
    :param scalar_labels:
        The samples' labels. Each row is a scalar label of the related sample (same row in "samples")
    :return:
        The trained NN.
    """

    train_samples = np.array(train_samples)
    scalar_labels = np.array(scalar_labels)
    num_samples = train_samples.shape[0]
    training_set_size = round(num_samples * 0.7)

    training_set = {
        'samples': train_samples[:training_set_size, :],
        'label': scalar_labels[:training_set_size]
    }
    validation_set = {
        'samples': train_samples[training_set_size + 1:, :],
        'label': scalar_labels[training_set_size + 1:]
    }

    # Learning
    rprop = RPROP(NN.get_num_params())
    back_prop = bp.BackPropagation(NN)
    evaluated_net_config_list = []
    for epoch in range(max_epoch):
        print("Run epoch: " + str(epoch + 1) + "...", end="")
        start = time.time()

        evaluated_net_config = _run_epoch(NN, rprop, back_prop, training_set, validation_set)
        evaluated_net_config_list.append(evaluated_net_config)

        end = time.time()
        timing = str(round(end - start, 2))
        print(" Complete. (" + timing + "s) ")

    # Retrieve best epoch (smallest validation error)
    all_validation_errors = [config.validation_error for config in evaluated_net_config_list]
    all_validation_errors = np.array(all_validation_errors)
    best_epoch = np.argmin(all_validation_errors)

    # Update NN with best parameters
    NN.set_all_params(evaluated_net_config_list[best_epoch].net_params)

    _plot_errors(evaluated_net_config_list)

    return NN


def _run_epoch(NN, rprop, back_prop, training_set, validation_set):
    """
    This method computes the epoch.

    :param NN:
        The NN.
    :param rprop:
        The RPROP object.
    :param back_prop:
        The back propagation object.
    :param training_set:
        The training set's samples. This input must be a matrix where each row is a sample and each column is a feature.
    :param validation_set:
        The validation set's samples. This input must be a matrix where each row is a sample and each column is a feature.
    :param num_params:
        The number of parameters.
    :return:
        The evaluation of the network's configuration.
    """

    grad_E_tot = np.zeros(NN.get_num_params())
    training_set_size = training_set['samples'].shape[0]

    for i in range(training_set_size):
        grad_E_sample = back_prop.compute_error_gradient(training_set['samples'][i],
                                                         np.array([training_set['label'][i]]))
        grad_E_tot += grad_E_sample

    # Update NN parameters
    rprop.update(NN, grad_E_tot)

    # Compute errors
    train_error = _compute_error(NN, training_set['samples'], training_set['label'])
    val_error = _compute_error(NN, validation_set['samples'], validation_set['label'])

    # Save the evaluated NN's information of this epoch
    evaluated_net_config = EvaluatedNetConfig(NN.get_all_params(), train_error, val_error)

    return evaluated_net_config


def _plot_errors(net_config_evaluated_list):
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
    plt.ylim([0.0, 1.0])

    plt.show()


def binary_accuracy(NN, samples, scalar_labels, threshold):
    """
    This function computes the accuracy of a NN on a specific dataset.

    :param NN:
        The NN to be evaluated.
    :param samples:
        The dataset's samples. This input must be a matrix where each row is a sample and each column is a feature.
    :param scalar_labels:
        The samples' labels. Each row is the scalar label of the related sample (same row in "samples").
    :param threshold:
        The minimum value to set to 1 the prediction.
    :return:
        The accuracy as a percentage.
    """

    num_correct = 0
    num_samples = samples.shape[0]
    for i in range(num_samples):
        all_output = NN.compute_network(samples[i])[1]
        output_net = all_output[-1][0]

        # Retrieve the most probable class
        if output_net >= threshold:
            predict_class = 1
        else:
            predict_class = 0

        # Compare
        target = scalar_labels[i]
        if predict_class == target:
            num_correct += 1

    acc = (num_correct / num_samples) * 100

    return acc
