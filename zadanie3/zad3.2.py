import numpy as np

X_AND = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_AND = np.array([0, 0, 0, 1])

X_NOT = np.array([[0], [1]])
y_NOT = np.array([1, 0])

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 1, 0])

X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_XOR = np.array([0, 1, 1, 0])

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.random.rand(n_features)
        self.bias = np.random.rand()

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)

    def XOR_perceptron(x1, x2):
        x1_NOT = NOT_perceptron.predict(np.array([x1]))
        x2_NOT = NOT_perceptron.predict(np.array([x2]))

        x1_AND_NOT_x2 = AND_perceptron.predict(np.array([x1, x2_NOT]))
        NOT_x1_AND_x2 = AND_perceptron.predict(np.array([x1_NOT, x2]))

        y_pred = OR_perceptron.predict(np.array([x1_AND_NOT_x2, NOT_x1_AND_x2]))
        return y_pred


AND_perceptron = Perceptron()
AND_perceptron.fit(X_AND, y_AND)

print("Perceptron dla fukcji AND:")
for x_i, y_true in zip(X_AND, y_AND):
    print(f"Wejście: {x_i}, Wyjście: {y_true}")

NOT_perceptron = Perceptron()
NOT_perceptron.fit(X_NOT, y_NOT)

print("\nPerceptron dla fukcji NOT:")
for x_i, y_true in zip(X_NOT, y_NOT):
    print(f"Wejście: {x_i}, Wyjście: {y_true}")

x1_AND_NOT_x2_perceptron = Perceptron()
x1_AND_NOT_x2_perceptron.fit(X, y)

print("\nPerceptron dla funkcji x1 ∧ ¬x2:")
for x_i, y_true in zip(X, y):
    print(f"Wejście: {x_i}, Wyjście: {y_true}")

XOR_perceptron = Perceptron()
XOR_perceptron.fit(X_XOR, y_XOR)

print("\nDwuwarstwowa sieć perceptronów dla funkcji XOR:")
for x_i, y_true in zip(X_XOR, y_XOR):
    print(f"Wejście: {x_i}, Wyjście: {y_true}")