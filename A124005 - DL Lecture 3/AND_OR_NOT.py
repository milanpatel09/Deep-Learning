#perceptron _for_AND_OR_NOT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# Define the logic gate data
logic_gates = {
    "AND": {"X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), "y": np.array([0, 0, 0, 1])},
    "OR": {"X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), "y": np.array([0, 1, 1, 1])},
    "NOT": {"X": np.array([[0], [1]]), "y": np.array([1, 0])},
}

def train_perceptron(X, y):
    model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
    model.fit(X, y)
    return model

# Train and test the perceptron for each logic gate
for gate, data in logic_gates.items():
    print(f"Training Perceptron for {gate} gate")
    model = train_perceptron(data["X"], data["y"])
    predictions = model.predict(data["X"])
    print(f"Predictions for {gate} gate: {predictions}")
    print("-" * 30)