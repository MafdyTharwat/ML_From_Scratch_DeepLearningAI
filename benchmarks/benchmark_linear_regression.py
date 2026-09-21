import time

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ml_from_scratch.linear_models.linear_regression import LinearRegression

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Training time : {training_time:.6f} seconds")
    print(f"MSE           : {mse:.6f}")
    print(f"RMSE          : {rmse:.6f}")
    print(f"MAE           : {mae:.6f}")
    print(f"R²            : {r2:.6f}")

    return {
        "training_time": training_time,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": y_pred,
    }


def main():
    X, y = make_regression(
        n_samples = 1000,
        n_features = 5,
        n_informative = 5,
        noise = 10,
        random_state = 42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 42,
    )

    # Our implementation
    our_model = LinearRegression(
        learning_rate = 0.01,
        n_iterations = 5000,
        tolerance = 1e-8,
    )

    our_results = evaluate_model(
        "My Linear Regression",
        our_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    sklearn_model = SklearnLinearRegression()

    sklearn_results = evaluate_model(
        "scikit-learn Linear Regression",
        sklearn_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    prediction_difference = np.mean(
        np.abs(
            our_results["predictions"]
            - sklearn_results["predictions"]
        )
    )

    print("-" * 40)
    print(
        f"Mean absolute prediction difference: "
        f"{prediction_difference:.6f}"
    )


if __name__ == "__main__":
    main()