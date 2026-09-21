import numpy as np

from ml_from_scratch.neural_networks.activations import (
    ReLU,
    Sigmoid,
    Tanh,
)


class TestReLU:

    def test_forward(self):
        activation = ReLU()

        X = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        output = activation.forward(X)

        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])

        np.testing.assert_allclose(output, expected)

    def test_backward(self):
        activation = ReLU()

        X = np.array([-2.0, 0.0, 2.0])
        activation.forward(X)

        doutput = np.array([1.0, 1.0, 1.0])

        dinput = activation.backward(doutput)

        expected = np.array([0.0, 0.0, 1.0])

        np.testing.assert_allclose(dinput, expected)

    def test_forward_preserves_shape(self):
        activation = ReLU()

        X = np.random.randn(10, 5)

        output = activation.forward(X)

        assert output.shape == X.shape


class TestSigmoid:

    def test_forward(self):
        activation = Sigmoid()

        X = np.array([-1.0, 0.0, 1.0])

        output = activation.forward(X)

        expected = 1 / (1 + np.exp(-X))

        np.testing.assert_allclose(output, expected)

    def test_output_range(self):
        activation = Sigmoid()

        X = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])

        output = activation.forward(X)

        assert np.all(output >= 0)
        assert np.all(output <= 1)

    def test_backward(self):
        activation = Sigmoid()

        X = np.array([-1.0, 0.0, 1.0])
        output = activation.forward(X)

        doutput = np.ones_like(X)

        dinput = activation.backward(doutput)

        expected = output * (1 - output)

        np.testing.assert_allclose(dinput, expected)

    def test_forward_preserves_shape(self):
        activation = Sigmoid()

        X = np.random.randn(10, 5)

        output = activation.forward(X)

        assert output.shape == X.shape


class TestTanh:

    def test_forward(self):
        activation = Tanh()

        X = np.array([-1.0, 0.0, 1.0])

        output = activation.forward(X)

        expected = np.tanh(X)

        np.testing.assert_allclose(output, expected)

    def test_output_range(self):
        activation = Tanh()

        X = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])

        output = activation.forward(X)

        assert np.all(output >= -1)
        assert np.all(output <= 1)

    def test_backward(self):
        activation = Tanh()

        X = np.array([-1.0, 0.0, 1.0])
        output = activation.forward(X)

        doutput = np.ones_like(X)

        dinput = activation.backward(doutput)

        expected = 1 - output ** 2

        np.testing.assert_allclose(dinput, expected)

    def test_forward_preserves_shape(self):
        activation = Tanh()

        X = np.random.randn(10, 5)

        output = activation.forward(X)

        assert output.shape == X.shape