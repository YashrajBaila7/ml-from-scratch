from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def update(self, layer) -> None:
        pass


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01): self.lr = lr

    def update(self, layer) -> None:
        if layer.w is not None:
            layer.w -= self.lr * layer.dw
            layer.b -= self.lr * layer.db


class Adagrad(Optimizer):
    def __init__(self, lr: float = 0.01, eps: float = 1e-8):
        self.lr, self.eps = lr, eps
        self.g_w, self.g_b = {}, {}

    def update(self, layer) -> None:
        if layer.w is None:
            return
        l_id = id(layer)
        if l_id not in self.g_w:
            self.g_w[l_id] = np.zeros_like(layer.w)
            self.g_b[l_id] = np.zeros_like(layer.b)
        self.g_w[l_id] += layer.dw ** 2
        self.g_b[l_id] += layer.db ** 2
        layer.w -= (self.lr / (np.sqrt(self.g_w[l_id]) + self.eps)) * layer.dw
        layer.b -= (self.lr / (np.sqrt(self.g_b[l_id]) + self.eps)) * layer.db


class RMSprop(Optimizer):
    def __init__(self, lr: float = 0.001, beta: float = 0.9, eps: float = 1e-8):
        self.lr, self.beta, self.eps = lr, beta, eps
        self.v_w, self.v_b = {}, {}

    def update(self, layer) -> None:
        if layer.w is None:
            return
        l_id = id(layer)
        if l_id not in self.v_w:
            self.v_w[l_id] = np.zeros_like(layer.w)
            self.v_b[l_id] = np.zeros_like(layer.b)
        self.v_w[l_id] = self.beta * self.v_w[l_id] + \
            (1 - self.beta) * (layer.dw ** 2)
        self.v_b[l_id] = self.beta * self.v_b[l_id] + \
            (1 - self.beta) * (layer.db ** 2)
        layer.w -= (self.lr / (np.sqrt(self.v_w[l_id]) + self.eps)) * layer.dw
        layer.b -= (self.lr / (np.sqrt(self.v_b[l_id]) + self.eps)) * layer.db


class Adam(Optimizer):
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m_w, self.m_b = {}, {}
        self.v_w, self.v_b = {}, {}
        self.t = {}

    def update(self, layer) -> None:
        if layer.w is None:
            return
        l_id = id(layer)
        if l_id not in self.m_w:
            self.m_w[l_id], self.m_b[l_id] = np.zeros_like(
                layer.w), np.zeros_like(layer.b)
            self.v_w[l_id], self.v_b[l_id] = np.zeros_like(
                layer.w), np.zeros_like(layer.b)
            self.t[l_id] = 0
        self.t[l_id] += 1
        t = self.t[l_id]
        self.m_w[l_id] = self.beta1 * \
            self.m_w[l_id] + (1 - self.beta1) * layer.dw
        self.m_b[l_id] = self.beta1 * \
            self.m_b[l_id] + (1 - self.beta1) * layer.db
        self.v_w[l_id] = self.beta2 * self.v_w[l_id] + \
            (1 - self.beta2) * (layer.dw ** 2)
        self.v_b[l_id] = self.beta2 * self.v_b[l_id] + \
            (1 - self.beta2) * (layer.db ** 2)
        m_w_c = self.m_w[l_id] / (1 - self.beta1 ** t)
        m_b_c = self.m_b[l_id] / (1 - self.beta1 ** t)
        v_w_c = self.v_w[l_id] / (1 - self.beta2 ** t)
        v_b_c = self.v_b[l_id] / (1 - self.beta2 ** t)
        layer.w -= (self.lr / (np.sqrt(v_w_c) + self.eps)) * m_w_c
        layer.b -= (self.lr / (np.sqrt(v_b_c) + self.eps)) * m_b_c


class AdamW(Optimizer):
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m_w, self.m_b = {}, {}
        self.v_w, self.v_b = {}, {}
        self.t = {}

    def update(self, layer) -> None:
        if layer.w is None:
            return
        l_id = id(layer)
        if l_id not in self.m_w:
            self.m_w[l_id], self.m_b[l_id] = np.zeros_like(
                layer.w), np.zeros_like(layer.b)
            self.v_w[l_id], self.v_b[l_id] = np.zeros_like(
                layer.w), np.zeros_like(layer.b)
            self.t[l_id] = 0

        self.t[l_id] += 1
        t = self.t[l_id]

        # 1. Apply decoupled Weight Decay first (The "W" in AdamW)
        layer.w -= self.lr * self.weight_decay * layer.w

        # 2. Standard Adam moment updates
        self.m_w[l_id] = self.beta1 * \
            self.m_w[l_id] + (1 - self.beta1) * layer.dw
        self.m_b[l_id] = self.beta1 * \
            self.m_b[l_id] + (1 - self.beta1) * layer.db

        self.v_w[l_id] = self.beta2 * self.v_w[l_id] + \
            (1 - self.beta2) * (layer.dw ** 2)
        self.v_b[l_id] = self.beta2 * self.v_b[l_id] + \
            (1 - self.beta2) * (layer.db ** 2)

        m_w_c = self.m_w[l_id] / (1 - self.beta1 ** t)
        m_b_c = self.m_b[l_id] / (1 - self.beta1 ** t)
        v_w_c = self.v_w[l_id] / (1 - self.beta2 ** t)
        v_b_c = self.v_b[l_id] / (1 - self.beta2 ** t)

        layer.w -= (self.lr / (np.sqrt(v_w_c) + self.eps)) * m_w_c
        layer.b -= (self.lr / (np.sqrt(v_b_c) + self.eps)) * m_b_c
