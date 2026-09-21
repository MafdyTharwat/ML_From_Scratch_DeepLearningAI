import numpy as np

from ml_from_scratch.neural_networks.neural_network import NeuralNetwork


class TestNeuralNetwork:

    def test_initialization(self):
        model = NeuralNetwork(input_size = 4, hidden_size = 5)

        assert len(model.layers) == 4
        assert model.loss is not None

    def test_forward_shape(self):
        model = NeuralNetwork(input_size = 4, hidden_size = 5)

        X = np.random.randn(10, 4)

        output = model.forward(X)

        assert output.shape == (10, 1)

    def test_forward_output_range(self):
        model = NeuralNetwork(input_size = 4, hidden_size = 5)

        X = np.random.randn(10, 4)

        output = model.forward(X)

        assert np.all(output >= 0)
        assert np.all(output <= 1)

    def test_predict_shape(self):
        model = NeuralNetwork(input_size = 4, hidden_size = 5)

        X = np.random.randn(10, 4)

        predictions = model.predict(X)

        assert predictions.shape == (10, 1)

    def test_predict_contains_binary_values(self):
        model = NeuralNetwork(input_size = 4, hidden_size = 5) 

        X = np.random.randn(10, 4)

        predictions = model.predict(X)

        assert np.all(np.isin(predictions, [0, 1]))

    def test_training_reduces_loss(self):
        np.random.seed(22)

        X = np.random.randn(100, 2)

        y = ((X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1))

        model = NeuralNetwork(input_size = 2, hidden_size = 5, learning_rate = 0.1, n_iterations = 1000)

        model.fit(X, y)

        assert model.loss_history[-1] < model.loss_history[0]

    def test_loss_history(self):
        model = NeuralNetwork(input_size = 2, hidden_size = 5, n_iterations = 100)

        X = np.random.randn(20, 2)
        y = np.random.randint(0, 2, size = (20, 1))

        model.fit(X, y)

        assert len(model.loss_history) == 100

    def test_score(self):
        np.random.seed(22)

        X = np.random.randn(50, 2)

        y = ((X[:, 0] > 0).astype(int))

        model = NeuralNetwork(input_size = 2, hidden_size = 5, learning_rate = 0.1, n_iterations = 500)

        model.fit(X, y)

        score = model.score(X, y)

        assert 0 <= score <= 1