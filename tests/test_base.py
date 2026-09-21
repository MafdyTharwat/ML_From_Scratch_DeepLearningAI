import pytest

from ml_from_scratch.base import BaseEstimator


class DummyEstimator(BaseEstimator):
    def __init__(self, alpha=1.0, beta=2):
        self.alpha = alpha
        self.beta = beta

        self._parameter_names = [
            "alpha",
            "beta",
        ]

        self.learned_state = None


def test_get_params():
    model = DummyEstimator(alpha=0.5, beta=10)

    params = model.get_params()

    assert params == {
        "alpha": 0.5,
        "beta": 10,
    }


def test_set_params():
    model = DummyEstimator()

    result = model.set_params(
        alpha=0.25,
        beta=5,
    )

    assert result is model
    assert model.alpha == 0.25
    assert model.beta == 5


def test_set_invalid_param():
    model = DummyEstimator()

    with pytest.raises(ValueError):
        model.set_params(gamma=10)


def test_get_params_does_not_include_model_state():
    model = DummyEstimator()
    model.learned_state = "something learned"

    params = model.get_params()

    assert "learned_state" not in params

from ml_from_scratch.base import clone

def test_clone_creates_new_estimator():
    model = DummyEstimator()

    cloned = clone(model)

    assert cloned is not model
    assert cloned.get_params() == model.get_params()

def test_clone_pipeline_creates_fresh_steps():
    from ml_from_scratch.pipeline.pipeline import Pipeline
    from ml_from_scratch.preprocessing.scaler import MyStandardScaler
    from ml_from_scratch.linear_models.linear_regression import LinearRegression

    scaler = MyStandardScaler()
    model = LinearRegression()

    pipeline = Pipeline([
        ("scaler", scaler),
        ("model", model),
    ])

    cloned_pipeline = clone(pipeline)

    assert cloned_pipeline is not pipeline

    assert (
        cloned_pipeline.steps[0][1]
        is not pipeline.steps[0][1]
    )

    assert (
        cloned_pipeline.steps[1][1]
        is not pipeline.steps[1][1]
    )