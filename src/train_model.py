"""
Model training module for the Stock Price Prediction System.

Responsibilities:
- Load preprocessed feature and target data
- Split the dataset into training and testing sets chronologically
- Create the Linear Regression model
- Train the model
- Save the trained model using Joblib

This file does not handle Streamlit UI or direct user predictions.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import (
    DEFAULT_TICKER,
    TEST_SIZE,
    create_required_directories,
    get_model_file_path,
)
from preprocess import get_preprocessed_data


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Split the dataset into training and testing sets.

    The split is chronological (not shuffled), since stock data is a
    time series and future rows must not leak into the training set.

    Args:
        features: Input feature DataFrame.
        target: Target Series containing tomorrow's closing price.

    Returns:
        A tuple containing:
        - X_train
        - X_test
        - y_train
        - y_test
    """

    split_index = int(len(features) * (1 - TEST_SIZE))

    X_train = features.iloc[:split_index]
    X_test = features.iloc[split_index:]

    y_train = target.iloc[:split_index]
    y_test = target.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def create_model() -> LinearRegression:
    """
    Create a Linear Regression model.

    Returns:
        An untrained LinearRegression model.
    """

    return LinearRegression()


def train_model(
    model: LinearRegression,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LinearRegression:
    """
    Train the Linear Regression model.

    Args:
        model: Untrained Linear Regression model.
        X_train: Training feature data.
        y_train: Training target values.

    Returns:
        The trained Linear Regression model.
    """

    model.fit(X_train, y_train)

    return model


def save_model(
    model: LinearRegression,
    model_path: Path,
) -> None:
    """
    Save the trained model to disk using Joblib.

    Args:
        model: Trained Linear Regression model.
        model_path: Path where the model will be saved.

    Raises:
        OSError: If the model cannot be saved.
    """

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_path)

    except OSError as error:
        raise OSError(
            f"Unable to save the trained model at: {model_path}"
        ) from error


def display_training_summary(
    model: LinearRegression,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """
    Display model training information.

    Args:
        model: Trained Linear Regression model.
        X_train: Training feature data.
        X_test: Testing feature data.
    """

    print("\nTraining Summary")
    print("-" * 50)
    print(f"Training records : {len(X_train)}")
    print(f"Testing records  : {len(X_test)}")
    print(f"Model type       : {type(model).__name__}")
    print(f"Model intercept  : {model.intercept_:.4f}")

    print("\nFeature Coefficients")
    print("-" * 50)

    for feature_name, coefficient in zip(
        X_train.columns,
        model.coef_,
    ):
        print(f"{feature_name:<20}: {coefficient:.4f}")


def execute_training_pipeline(
    ticker: str = DEFAULT_TICKER,
) -> tuple[
    LinearRegression,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    """
    Execute the complete model training workflow for a ticker.

    Steps:
    1. Create required folders
    2. Download and preprocess market data
    3. Split the dataset chronologically
    4. Create the Linear Regression model
    5. Train the model
    6. Save the trained model

    Args:
        ticker: Stock ticker symbol to train a model for.

    Returns:
        A tuple containing:
        - trained model
        - X_train
        - X_test
        - y_train
        - y_test
        - latest_row (most recent feature row, used for prediction)
    """

    create_required_directories()

    features, target, _history, latest_row = get_preprocessed_data(ticker)

    X_train, X_test, y_train, y_test = split_dataset(
        features,
        target,
    )

    model = create_model()

    trained_model = train_model(
        model,
        X_train,
        y_train,
    )

    save_model(trained_model, get_model_file_path(ticker))

    return (
        trained_model,
        X_train,
        X_test,
        y_train,
        y_test,
        latest_row,
    )


if __name__ == "__main__":
    try:
        (
            trained_model,
            X_train_data,
            X_test_data,
            y_train_data,
            y_test_data,
            latest_features,
        ) = execute_training_pipeline()

        display_training_summary(
            trained_model,
            X_train_data,
            X_test_data,
        )

        print("\nModel training completed successfully.")
        print(f"Model saved at: {get_model_file_path(DEFAULT_TICKER)}")

    except (ValueError, OSError) as error:
        print(f"Model training failed: {error}")
