from abc import ABC, abstractmethod
import numpy as np
from activation import Activation, Linear
from initializers import Initializer


class Layer(ABC):
    def __init__(self):
        self.w, self.b = None, None
        self.dw, self.db = None, None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: pass
    @abstractmethod
    def backward(self, d_out: np.ndarray) -> np.ndarray: pass


class DenseLayer(Layer):
    def __init__(self, nin: int, nout: int, activation: Activation = Linear(), init_method: str = "he"):
        super().__init__()

        if init_method == "he":
            self.w = Initializer.he(nin, nout)
        elif init_method == "xavier":
            self.w = Initializer.xavier(nin, nout)
        else:
            self.w = np.random.normal(0.0, 0.01, size=(nin, nout))

        self.b = np.zeros((1, nout))
        self.activation = activation
        self.last_x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        return self.activation(x @ self.w + self.b)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        d_linear = self.activation.backward(d_out)
        self.dw = self.last_x.T @ d_linear
        self.db = np.sum(d_linear, axis=0, keepdims=True)
        return d_linear @ self.w.T


class DropoutLayer(Layer):
    def __init__(self, drop_rate: float):
        super().__init__()
        self.drop_rate = drop_rate
        self.keep_prob = 1.0 - drop_rate
        self.mask = None
        self.is_training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.is_training or self.drop_rate == 0.0:
            return x
        self.mask = np.random.binomial(
            1, self.keep_prob, size=x.shape) / self.keep_prob
        return x * self.mask

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if not self.is_training or self.drop_rate == 0.0:
            return d_out
        return d_out * self.mask
