import numpy as np

alpha = 0.02

activation_function = {
    'relu': (lambda x: np.maximum(0, x)),
    'identity': (lambda x: x),
    'leakyrelu': (lambda x: x if x > 0 else alpha*x),
    'sigmoid': (lambda x: 1/(1+np.exp(-x)))
}

activation_function_der = {
    'relu': (lambda x: 0 if x < 0 else 1),
    'identity': (lambda x: 1),
    'leakyrelu': (lambda x: 1 if x > 0 else alpha),
    'sigmoid': (lambda x: activation_function['sigmoid'](x) * (1 - activation_function['sigmoid'](x)))
}
