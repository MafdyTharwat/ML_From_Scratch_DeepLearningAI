import numpy as np


class MSELoss:
    def forward(self, y, y_hat):
        y = np.asarray(y, dtype = float)
        y_hat = np.asarray(y_hat, dtype = float)

        if y.shape != y_hat.shape:
            raise ValueError('y and predicted y must have the same shape')

        self.y = y
        self.y_hat = y_hat

        return np.mean((y - y_hat) ** 2)

    def backward(self):
        n = self.y.size

        return (2 / n) * (self.y_hat - self.y)


class BinaryCrossEntropyLoss:
    def forward(self, y, y_hat):
        y = np.asarray(y, dtype = float)
        y_hat = np.asarray(y_hat, dtype = float)

        if y.shape != y_hat.shape:
            raise ValueError('y and predicted y must have the same shape')

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError('y must contain only binary values (0 and 1)')

        if np.any((y_hat < 0) | (y_hat > 1)):
            raise ValueError('predicted y must contain probabilities between (0 and 1)')

        self.y = y
        self.y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)

        return -np.mean(
            self.y * np.log(self.y_hat)
            + (1 - self.y) * np.log(1 - self.y_hat)
        )

    def backward(self):
        n = self.y.size

        return (
            (self.y_hat - self.y)
            / (self.y_hat * (1 - self.y_hat))
        ) / n