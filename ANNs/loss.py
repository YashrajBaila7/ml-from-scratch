from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.forward(y_true, y_pred)

    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @abstractmethod
    def get_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class MSE(Loss):
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.square(y_true - y_pred)))

    def get_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (2 / y_true.shape[0]) * (y_pred - y_true)


class BinaryCrossEntropy(Loss):
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    def get_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (1 / y_true.shape[0]) * ((y_pred - y_true) / (y_pred * (1 - y_pred)))


class CategoricalCrossEntropy(Loss):
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.sum(y_true * np.log(y_pred)) / y_true.shape[0])

    def get_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / y_pred) / y_true.shape[0]
