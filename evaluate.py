"""
Model evaluation module for the Stock Price Prediction System.

Responsibilities:
- Load and preprocess market data for a ticker
- Split the dataset using the same configuration as training
- Train a Linear Regression model on the training portion
- Generate predictions for the testing dataset
- Calculate evaluation metrics
- Display actual and predicted closing prices

This file does not save the model to disk.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from config import DEFAULT_TICKER
from preprocess import get_preprocessed_data
from train_model import create_model, split_dataset, train_model


@dataclass(frozen=True)
class EvaluationResult:
    """
    Store the calculated model evaluation metrics.

    Attributes:
        mean_absolute_error: Average absolute prediction error, in price units.
        mean_squared_error: Average squared prediction error.
        root_mean_squared_error: Square root of MSE, in price units.
        r2_score: Percentage of target variation explained by the model.
    """

    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    r2_score: float


def generate_predictions(
    model: LinearRegression,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Generate predictions for the testing dataset.

    Args:
        model: Trained Linear Regression model.
        X_test: Testing feature data.

    Returns:
        NumPy array containing predicted closing prices.
    """

    predictions = model.predict(X_test)

    return np.asarray(predictions, dtype=float)


def calculate_metrics(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> EvaluationResult:
    """
    Calculate regression evaluation metrics.

    Metrics:
    - MAE
    - MSE
    - RMSE
    - R² score

    Args:
        y_test: Actual closing prices from the testing dataset.
        predictions: Closing prices predicted by the model.

    Returns:
        EvaluationResult containing all calculated metrics.
    """

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    return EvaluationResult(
        mean_absolute_error=float(mae),
        mean_squared_error=float(mse),
        root_mean_squared_error=float(rmse),
        r2_score=float(r2),
    )


def create_comparison_table(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Create a table comparing actual and predicted closing prices.

    Args:
        y_test: Actual closing prices.
        predictions: Predicted closing prices.

    Returns:
        DataFrame containing actual prices, predicted prices, and error,
        indexed by date.
    """

    comparison = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": np.round(predictions, 2),
        },
        index=y_test.index,
    )

    comparison["AbsoluteError"] = np.round(
        np.abs(comparison["Actual"] - comparison["Predicted"]),
        2,
    )

    return comparison


def display_evaluation_results(result: EvaluationResult) -> None:
    """
    Display the calculated model evaluation metrics.

    Args:
        result: Object containing evaluation metrics.
    """

    print("\nModel Evaluation Results")
    print("-" * 50)
    print(f"Mean Absolute Error (MAE)  : {result.mean_absolute_error:.4f}")
    print(f"Mean Squared Error (MSE)   : {result.mean_squared_error:.4f}")
    print(f"Root Mean Squared Error    : {result.root_mean_squared_error:.4f}")
    print(f"R² Score                   : {result.r2_score:.4f}")


def interpret_r2_score(score: float) -> str:
    """
    Return a simple interpretation of the R² score.

    This interpretation is intended for classroom explanation and should
    not be treated as a universal industry threshold. High R² values on
    stock data often reflect that tomorrow's price tracks closely with
    today's price, not genuine predictive skill.

    Args:
        score: Calculated R² score.

    Returns:
        Human-readable model performance interpretation.
    """

    if score >= 0.90:
        return "Excellent model fit."

    if score >= 0.75:
        return "Good model fit."

    if score >= 0.50:
        return "Moderate model fit."

    if score >= 0.0:
        return "Weak model fit. Improvement is required."

    return (
        "Poor model fit. The model performs worse than "
        "predicting the average target value."
    )


def execute_evaluation_pipeline(
    ticker: str = DEFAULT_TICKER,
) -> tuple[EvaluationResult, pd.DataFrame]:
    """
    Execute the complete model evaluation workflow for a ticker.

    Steps:
    1. Download and preprocess market data
    2. Split it chronologically, using the same settings as training
    3. Train a Linear Regression model on the training portion
    4. Generate test predictions
    5. Calculate evaluation metrics
    6. Build an actual-versus-predicted comparison table

    Args:
        ticker: Stock ticker symbol to evaluate.

    Returns:
        A tuple containing:
        - EvaluationResult
        - Actual-versus-predicted comparison DataFrame
    """

    features, target, _history, _latest_row = get_preprocessed_data(ticker)

    X_train, X_test, y_train, y_test = split_dataset(
        features,
        target,
    )

    model = train_model(create_model(), X_train, y_train)

    predictions = generate_predictions(model, X_test)

    evaluation_result = calculate_metrics(y_test, predictions)

    comparison_table = create_comparison_table(y_test, predictions)

    return evaluation_result, comparison_table


if __name__ == "__main__":
    try:
        results, comparison = execute_evaluation_pipeline()

        display_evaluation_results(results)

        print(
            "\nPerformance Interpretation: "
            f"{interpret_r2_score(results.r2_score)}"
        )

        print("\nActual vs Predicted Closing Price")
        print("-" * 50)
        print(comparison.tail(10).to_string())

        print("\nModel evaluation completed successfully.")

    except ValueError as error:
        print(f"Model evaluation failed: {error}")
