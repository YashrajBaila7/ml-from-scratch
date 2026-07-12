from typing import List
import numpy as np
from layers import Layer
from loss import Loss, CategoricalCrossEntropy
from activation import Softmax
from optimizer import Optimizer


class NeuralNetwork:
    def __init__(self, loss_fn: Loss):
        self.layers: List[Layer] = []
        self.loss_fn = loss_fn

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def train(self) -> None:
        """Sets all layers to training mode (e.g., enables dropout)."""
        for layer in self.layers:
            if hasattr(layer, 'is_training'):
                layer.is_training = True

    def eval(self) -> None:
        """Sets all layers to evaluation mode (e.g., disables dropout)."""
        for layer in self.layers:
            if hasattr(layer, 'is_training'):
                layer.is_training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, loss_grad: np.ndarray) -> None:
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_step(self, x: np.ndarray, y: np.ndarray, optimizer: Optimizer) -> float:
        y_pred = self(x)
        loss_val = self.loss_fn(y, y_pred)

        if isinstance(self.loss_fn, CategoricalCrossEntropy) and isinstance(self.layers[-1].w, np.ndarray) and isinstance(getattr(self.layers[-1], 'activation', None), Softmax):
            loss_grad = (y_pred - y) / x.shape[0]
            last_layer = self.layers[-1]
            last_layer.dw = last_layer.last_x.T @ loss_grad
            last_layer.db = np.sum(loss_grad, axis=0, keepdims=True)
            remaining_grad = loss_grad @ last_layer.w.T
            for layer in reversed(self.layers[:-1]):
                remaining_grad = layer.backward(remaining_grad)
        else:
            loss_grad = self.loss_fn.get_gradient(y, y_pred)
            self.backward(loss_grad)

        for layer in self.layers:
            optimizer.update(layer)

        return loss_val

    def save_model(self, filepath: str) -> None:
        """Saves weights and biases of all layers into a compressed numpy file."""
        weights_dict = {}
        for idx, layer in enumerate(self.layers):
            if layer.w is not None:
                weights_dict[f"w_{idx}"] = layer.w
                weights_dict[f"b_{idx}"] = layer.b
        np.savez_compressed(filepath, **weights_dict)
        print(f"Model successfully saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Loads and overwrites weights and biases from an npz file into layers."""
        data = np.load(filepath)
        for idx, layer in enumerate(self.layers):
            if layer.w is not None:
                if f"w_{idx}" in data and f"b_{idx}" in data:
                    layer.w = data[f"w_{idx}"]
                    layer.b = data[f"b_{idx}"]
                else:
                    raise KeyError(
                        f"Could not find weights for layer index {idx} in {filepath}")
        print(f"Model successfully loaded from {filepath}")
