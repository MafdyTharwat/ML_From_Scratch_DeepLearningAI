import numpy as np
import pytest

from ml_from_scratch.neural_networks.losses import (
    MSELoss,
    BinaryCrossEntropyLoss,
)


class TestMSELoss:

    def test_forward(self):
        loss = MSELoss()

        y = np.array([1.0, 2.0, 3.0])
        y_hat = np.array([1.0, 3.0, 2.0])

        result = loss.forward(y, y_hat)

        expected = (0 + 1 + 1) / 3

        assert result == pytest.approx(expected)

    def test_backward(self):
        loss = MSELoss()

        y = np.array([1.0, 2.0])
        y_hat = np.array([2.0, 4.0])

        loss.forward(y, y_hat)

        gradient = loss.backward()

        expected = np.array([1.0, 2.0])

        np.testing.assert_allclose(gradient, expected)

    def test_zero_loss(self):
        loss = MSELoss()

        y = np.array([1.0, 2.0, 3.0])
        y_hat = np.array([1.0, 2.0, 3.0])

        assert loss.forward(y, y_hat) == pytest.approx(0.0)

    def test_shape_mismatch(self):
        loss = MSELoss()

        y = np.array([1.0, 2.0])
        y_hat = np.array([1.0])

        with pytest.raises(ValueError):
            loss.forward(y, y_hat)


class TestBinaryCrossEntropyLoss:

    def test_forward(self):
        loss = BinaryCrossEntropyLoss()

        y = np.array([1.0, 0.0])
        y_hat = np.array([0.9, 0.1])

        result = loss.forward(y, y_hat)

        expected = -np.mean(np.log([0.9, 0.9]))

        assert result == pytest.approx(expected)

    def test_backward(self):
        loss = BinaryCrossEntropyLoss()

        y = np.array([1.0, 0.0])
        y_hat = np.array([0.8, 0.2])

        loss.forward(y, y_hat)

        gradient = loss.backward()

        expected = np.array([-0.625, 0.625])

        np.testing.assert_allclose(gradient, expected)

    def test_perfect_predictions_have_low_loss(self):
        loss = BinaryCrossEntropyLoss()

        y = np.array([1.0, 0.0])
        y_hat = np.array([1.0, 0.0])

        result = loss.forward(y, y_hat)

        assert result < 1e-10

    def test_invalid_targets(self):
        loss = BinaryCrossEntropyLoss()

        y = np.array([1.0, 2.0])
        y_hat = np.array([0.8, 0.2])

        with pytest.raises(ValueError):
            loss.forward(y, y_hat)

    def test_invalid_probabilities(self):
        loss = BinaryCrossEntropyLoss()

        y = np.array([1.0, 0.0])
        y_hat = np.array([1.2, 0.2])

        with pytest.raises(ValueError):
            loss.forward(y, y_hat)

    def test_shape_mismatch(self):
        loss = BinaryCrossEntropyLoss()

        y = np.array([1.0, 0.0])
        y_hat = np.array([0.8])

        with pytest.raises(ValueError):
            loss.forward(y, y_hat)