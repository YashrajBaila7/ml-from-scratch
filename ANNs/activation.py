import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod


class Activation(ABC):
    def __call__(self, z: ndarray) -> ndarray:
        return self.forward(z)

    @classmethod
    @abstractmethod
    def forward(self, z: ndarray) -> ndarray:
        pass

    @classmethod
    @abstractmethod
    def backward(self, d_out: ndarray) -> ndarray:
        pass


class Linear(Activation):
    def forward(self, z: ndarray) -> ndarray:
        return z

    def backward(self, d_out: ndarray) -> ndarray:
        return d_out


class ReLU(Activation):
    def __init__(self):
        self.last_z = None

    def forward(self, z: ndarray) -> ndarray:
        self.last_z = z
        return np.maximum(0, z)

    def backward(self, d_out: ndarray) -> ndarray:
        return d_out * (self.last_z > 0).astype(float)


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.last_z = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.last_z = z
        return np.where(z > 0, z, z * self.alpha)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        dx = np.ones_like(self.last_z)
        dx[self.last_z <= 0] = self.alpha
        return d_out * dx


class Sigmoid(Activation):
    def __init__(self): self.last_z = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.last_z = z
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        s = 1 / (1 + np.exp(-np.clip(self.last_z, -500, 500)))
        return d_out * s * (1 - s)


class Tanh(Activation):
    def __init__(self): self.last_z = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.last_z = z
        return np.tanh(z)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return d_out * (1 - np.square(np.tanh(self.last_z)))


class Softplus(Activation):
    def __init__(self): self.last_z = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.last_z = z
        return np.log(1 + np.exp(np.clip(z, -500, 500)))

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        s = 1 / (1 + np.exp(-np.clip(self.last_z, -500, 500)))
        return d_out * s


class Softmax(Activation):
    def __init__(self): self.last_out = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        shift_z = z - np.max(z, axis=-1, keepdims=True)
        exps = np.exp(shift_z)
        self.last_out = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.last_out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(d_out)
        for i, (out, dout) in enumerate(zip(self.last_out, d_out)):
            diag = np.diag(out)
            outer = np.outer(out, out)
            jacobian = diag - outer
            grad[i] = dout @ jacobian
        return grad
