import numpy as np 

def he_initialization(n_in, n_out):
    stddev = np.sqrt(2.0 / n_in)
    W = np.random.randn(n_out, n_in) * stddev 
    return W

def relu(x):
    return np.maximum(0, x)

def relu_derivative(z):
    return (z > 0).astype(float) 

def softmax(x):
    exp_x = np.exp(x) 
    return exp_x / np.sum(exp_x, axis=0, keepdims=True) 

def forward_propagation(x, weights, biases):
    a = x
    activations = [a] # Just input layer for now
    zs = []

    for i in range(1, len(weights)):
        z = weights[i] @ a + biases[i]
        if i < len(weights) - 1:
            a = relu(z)
        else:
            a = softmax(z)
        activations.append(a)
        zs.append(z)
    
    return activations[1], activations[2], activations[3], zs[0], zs[1], zs[2]

def back_propagation(x, y, weights, biases, cache):
    A1, A2, A3, Z1, Z2, Z3 = cache 
    m = x.shape[1] 
    
    W1 = weights[1]
    W2 = weights[2]
    W3 = weights[3]

    b1 = biases[1]
    b2 = biases[2]
    b3 = biases[3] 

    # Error of output layer 
    dLdZ3 = cross_entropy_loss_gradient(A3, y)
    dLdW3 = (dLdZ3 @ A2.T) / m
    dLdb3 = np.sum(dLdZ3, axis=1, keepdims=True) / m # we use axis=1, keepdims=True to sum over all samples. (Review)

    # Error of hidden layer 2
    dLdZ2 = W3.T @ dLdZ3 * relu_derivative(Z2)
    dLdW2 = (dLdZ2 @ A1.T) / m
    dLdb2 = np.sum(dLdZ2, axis=1, keepdims=True) / m 


    # Error of hidden layer 1
    dLdZ1 = W2.T @ dLdZ2 * relu_derivative(Z1)
    dLdW1 = (dLdZ1 @ x.T) / m 
    dLdb1 = np.sum(dLdZ1, axis=1, keepdims=True) / m 

    return dLdW1, dLdb1, dLdW2, dLdb2, dLdW3, dLdb3


def cross_entropy_loss(y_hat, y_true):
    assert y_hat.shape == y_true.shape
    
    epsilon = 1e-12
    y_hat = np.clip(y_hat, epsilon, 1.0 - epsilon)

    cross_entropy = -np.sum(y_true * np.log(y_hat)) / y_true.shape[1]

    return cross_entropy

def cross_entropy_loss_gradient(y_hat, y_true):
    assert y_hat.shape == y_true.shape
    
    return (y_hat - y_true) 
