import numpy as np

from ml_from_scratch.neural_networks.layers import Dense


class TestDenseInitialization:

    def test_weights_shape(self):
        layer = Dense(input_size = 4, output_size = 3)

        assert layer.weights.shape == (4, 3)

    def test_bias_shape(self):
        layer = Dense(input_size = 4, output_size = 3)

        assert layer.bias.shape == (1, 3)

    def test_weights_are_initialized(self):
        layer = Dense(input_size = 4, output_size = 3)

        assert layer.weights is not None

    def test_bias_is_initialized(self):
        layer = Dense(input_size = 4, output_size = 3)

        assert layer.bias is not None

    def test_initial_gradients_are_none(self):
        layer = Dense(input_size = 4, output_size = 3)

        assert layer.input is None
        assert layer.dweights is None
        assert layer.dbias is None


class TestDenseForward:

    def test_forward_output_shape(self):
        layer = Dense(input_size = 4, output_size = 3)

        X = np.random.randn(10, 4)

        output = layer.forward(X)

        assert output.shape == (10, 3)

    def test_forward_output_is_numpy_array(self):
        layer = Dense(input_size = 4, output_size = 3)

        X = np.random.randn(10, 4)

        output = layer.forward(X)

        assert isinstance(output, np.ndarray)

    def test_forward_stores_input(self):
        layer = Dense(input_size = 4, output_size = 3)

        X = np.random.randn(10, 4)

        layer.forward(X)

        assert layer.input is not None
        np.testing.assert_array_equal(layer.input, X)

    def test_forward_computation(self):
        layer = Dense(input_size = 2, output_size = 2)

        layer.weights = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])

        layer.bias = np.array([
            [5.0, 6.0]
        ])

        X = np.array([
            [1.0, 2.0]
        ])

        output = layer.forward(X)

        expected = np.array([
            [12.0, 16.0]
        ])

        np.testing.assert_allclose(output, expected)


class TestDenseBackward:

    def test_backward_input_gradient_shape(self):
        layer = Dense(input_size = 4, output_size = 3)

        X = np.random.randn(10, 4)

        output = layer.forward(X)

        doutput = np.random.randn(10, 3)

        dinput = layer.backward(doutput)

        assert dinput.shape == (10, 4)

    def test_backward_weight_gradient_shape(self):
        layer = Dense(input_size = 4, output_size = 3)

        X = np.random.randn(10, 4)

        layer.forward(X)

        doutput = np.random.randn(10, 3)

        layer.backward(doutput)

        assert layer.dweights.shape == (4, 3)

    def test_backward_bias_gradient_shape(self):
        layer = Dense(input_size = 4, output_size = 3)

        X = np.random.randn(10, 4)

        layer.forward(X)

        doutput = np.random.randn(10, 3)

        layer.backward(doutput)

        assert layer.dbias.shape == (1, 3)

    def test_backward_gradients_are_numpy_arrays(self):
        layer = Dense(input_size = 4, output_size = 3)

        X = np.random.randn(10, 4)

        layer.forward(X)

        doutput = np.random.randn(10, 3)

        dinput = layer.backward(doutput)

        assert isinstance(dinput, np.ndarray)
        assert isinstance(layer.dweights, np.ndarray)
        assert isinstance(layer.dbias, np.ndarray)

    def test_backward_computation(self):
        layer = Dense(input_size = 2, output_size = 2)

        layer.weights = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])

        layer.bias = np.array([
            [0.0, 0.0]
        ])

        X = np.array([
            [1.0, 2.0]
        ])

        layer.forward(X)

        doutput = np.array([
            [5.0, 6.0]
        ])

        dinput = layer.backward(doutput)

        expected_dweights = np.array([
            [5.0, 6.0],
            [10.0, 12.0]
        ])

        expected_dbias = np.array([
            [5.0, 6.0]
        ])

        expected_dinput = np.array([
            [17.0, 39.0]
        ])

        np.testing.assert_allclose(
            layer.dweights,
            expected_dweights
        )

        np.testing.assert_allclose(
            layer.dbias,
            expected_dbias
        )

        np.testing.assert_allclose(
            dinput,
            expected_dinput
        )