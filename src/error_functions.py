import numpy as np


def cross_entropy_loss(output_lines, targets):

    output_lines = np.column_stack([output_lines])
    targets = np.column_stack([targets])

    # Check to avoid out-of-range
    if output_lines.size != targets.size:
        print("Output is not compatible with targets.")
        return -1

    # Compute the summation
    num_of_lines = output_lines.size
    summation = 0
    for line in range(num_of_lines):
        output = np.max([output_lines[line][0], 1e-7])  # To avoid 0
        summation += targets[line][0] * np.log(output)

    loss = -summation
    return loss


def cross_entropy_loss_per_sample(output, target):
    """
    :param output:
        The predicted target (scalar).
    :param target:
        The true target (scalar).
    :return:
        The cross entropy loss.
    """
    output = max(output, 1e-7)
    loss = target * np.log(output) + (1 - target) * np.log(1 - output)
    return -loss


def softmax(output_lines):
    exps = np.exp(output_lines - np.max(output_lines))
    return exps/np.sum(exps)


def cross_entropy_soft_max_der(soft_max_output, target):
    if target != 0 and target != 1:
        print("The target is not in one-hot encoding.")
        return -1
    der = soft_max_output - target
    return der


def cross_entropy_derivative(output, target):
    der = -((target/output) - ((1-target)/(1-output)))
    return der

