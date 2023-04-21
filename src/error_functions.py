import numpy as np


def cross_entropy_loss(output_lines, targets):
    output_lines = np.array(output_lines)
    targets = np.array(targets)

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
        output = np.max([output_lines[line][0], 10 ** (-5)])  # To avoid 0
        summation += targets[line][0] * np.log2(output)

    loss = -summation
    return loss


def softmax(output_lines):
    exps = np.exp(output_lines - np.max(output_lines))
    return exps/np.sum(exps)


def cross_entropy_soft_max_der(soft_max_output, target):
    if target != 0 and target != 1:
        print("The target is not in one-hot encoding.")
        return -1
    der = soft_max_output - target
    return der


