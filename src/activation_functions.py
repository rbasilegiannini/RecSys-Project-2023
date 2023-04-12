import numpy as np


activation_function = {
    'relu': (lambda x: np.maximum(0, x)),
    'identity': (lambda x: x)
}

activation_function_der = {
    'relu': (lambda x: 0 if x < 0 else 1),
    'identity': (lambda x: 1)
}
