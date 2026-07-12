import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        self.std[self.std == 0] = 1e-8

        return (X - self.mean) / self.std

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet!")
        return (X - self.mean) / self.std


def train_test_split(X: np.ndarray, Y: np.ndarray, test_size: float = 0.2, random_state: int = None):
    """Randomly shuffles and splits arrays into training and testing sets."""
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    split_idx = int(X.shape[0] * (1.0 - test_size))

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]


def get_mini_batches(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0], batch_size):
        batch_idx = indices[start_idx:start_idx + batch_size]
        yield X[batch_idx], y[batch_idx]
