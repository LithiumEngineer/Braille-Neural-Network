import numpy as np 
from utils import he_initialization, forward_propagation, back_propagation, cross_entropy_loss
from load_dataset import load_dataset

# First layer is size 28 x 28 = 784
# Two hidden layers of size 128 and 64. 
# Output layer of size 26 for each character from a-z. 
layer_sizes = [784, 128, 64, 26] 

# Hyperparameters 
learning_rate = 0.01
epochs = 1000

x, y = load_dataset('data/braille_dataset')

weights = [None] # Prepend an empty spot so weights[i] correspond to weights going into layer i
biases = [None] # Prepend an empty slot so weights[i] correspond to weights going into layer i

for i in range(len(layer_sizes) - 1):
    w = he_initialization(layer_sizes[i], layer_sizes[i + 1])
    b = np.zeros((layer_sizes[i + 1], 1))
    weights.append(w)
    biases.append(b)

for epoch in range(epochs):
    A1, A2, A3, Z1, Z2, Z3 = forward_propagation(x, weights, biases)
    cache = (A1, A2, A3, Z1, Z2, Z3)
    dLdW1, dLdb1, dLdW2, dLdb2, dLdW3, dLdb3 = back_propagation(x, y, weights, biases, cache)

    weights[1] -= learning_rate * dLdW1
    biases[1]  -= learning_rate * dLdb1
    weights[2] -= learning_rate * dLdW2
    biases[2]  -= learning_rate * dLdb2
    weights[3] -= learning_rate * dLdW3
    biases[3]  -= learning_rate * dLdb3

    if epoch % 100 == 0:
        loss = cross_entropy_loss(A3, y)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

