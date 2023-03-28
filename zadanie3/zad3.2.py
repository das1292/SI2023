import numpy as np


def step(x):
    return np.where(x >= 0, 1, 0)


def perceptron_learn(X, y):
    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0

    for _ in range(1000):
        for i in range(n_samples):
            z = np.dot(X[i], w) + b
            a = step(z)
            error = y[i] - a
            w += error * X[i]
            b += error

    return w, b


def perceptron_network(x1, x2):
    w_hidden = np.array([[20, -20], [-20, 20]])
    b_hidden = np.array([-10, 30])

    w_output = np.array([20, 20])
    b_output = -30

    z_hidden = np.dot(w_hidden, np.array([x1, x2])) + b_hidden
    a_hidden = step(z_hidden)

    z_output = np.dot(w_output, a_hidden) + b_output
    a_output = step(z_output)

    return a_output


# AND
print("\nAND:")
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
w_and, b_and = perceptron_learn(X_and, y_and)
print("w_and = ", w_and)
print("b_and = ", b_and)

# NOT
print("\nNOT:")
X_not = np.array([[0], [1]])
y_not = np.array([1, 0])
w_not, b_not = perceptron_learn(X_not, y_not)
print("w_not = ", w_not)
print("b_not = ", b_not)

# XOR
print("\nXOR:")
print("1 XOR 1 =", perceptron_network(1, 1))
print("1 XOR 0 =", perceptron_network(1, 0))
print("0 XOR 1 =", perceptron_network(0, 1))
print("0 XOR 0 =", perceptron_network(0, 0))