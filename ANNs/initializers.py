import numpy as np


class Initializer:
    @staticmethod
    def xavier(nin: int, nout: int) -> np.ndarray:
        """Xavier/Glorot initialization for Sigmoid/Tanh."""
        limit = np.sqrt(6.0 / (nin + nout))
        return np.random.uniform(-limit, limit, size=(nin, nout))

    @staticmethod
    def he(nin: int, nout: int) -> np.ndarray:
        """He/Kaiming initialization for ReLU/LeakyReLU."""
        std = np.sqrt(2.0 / nin)
        return np.random.normal(loc=0.0, scale=std, size=(nin, nout))
