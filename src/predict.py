"""
Prediction module for the Stock Price Prediction System.

Responsibilities:
- Validate the requested ticker
- Download and preprocess market data
- Train the Linear Regression model on historical data
- Evaluate the model on held-out (most recent) historical data
- Predict tomorrow's closing price
- Return a clean, structured result for display

This file does not contain any Streamlit UI code.

Note:
Unlike the student performance model, a stock model is trained fresh
for each request rather than loaded from a fixed saved file. Market
data changes every day, so a model saved yesterday would be stale.
The model is still saved to disk after training (see train_model.py)
for inspection and reuse, but predictions always use freshly trained
data.
"""

from dataclasses import dataclass

import pandas as pd

from config import DATA_PERIOD
from evaluate import (
    EvaluationResult,
    calculate_metrics,
    create_comparison_table,
    generate_predictions,
)
from preprocess import get_preprocessed_data
from train_model import (
    create_model,
    save_model,
    split_dataset,
    train_model,
)
from config import get_model_file_path


@dataclass(frozen=True)
class PredictionResult:
    """
    Store the complete result of a stock price prediction request.

    Attributes:
        ticker: Normalized stock ticker symbol.
        current_price: Most recent closing price.
        predicted_price: Predicted closing price for the next trading day.
        price_change: Predicted price minus the current price.
        evaluation: Model evaluation metrics on held-out historical data.
        historical_prices: Full cleaned closing-price history (for charts).
        comparison_table: Actual-versus-predicted prices on the test set.
    """

    ticker: str
    current_price: float
    predicted_price: float
    price_change: float
    evaluation: EvaluationResult
    historical_prices: pd.Series
    comparison_table: pd.DataFrame


def predict_next_close(
    ticker: str,
    period: str = DATA_PERIOD,
) -> PredictionResult:
    """
    Predict a stock's next closing price.

    Steps:
    1. Download and preprocess market data for the ticker
    2. Split the data chronologically into training and testing sets
    3. Train a Linear Regression model on the training set
    4. Evaluate the model on the testing set
    5. Save the trained model to disk
    6. Predict tomorrow's closing price using the most recent data

    Args:
        ticker: Stock ticker symbol entered by the user.
        period: Historical data window to download (e.g. "5y").

    Returns:
        A PredictionResult containing the prediction, evaluation
        metrics, and supporting data for display.

    Raises:
        ValueError: If the ticker is invalid or not enough historical
            data is available to train a model.
    """

    features, target, history, latest_row = get_preprocessed_data(
        ticker,
        period,
    )

    validated_ticker = ticker.strip().upper()

    X_train, X_test, y_train, y_test = split_dataset(features, target)

    model = train_model(create_model(), X_train, y_train)

    save_model(model, get_model_file_path(validated_ticker))

    test_predictions = generate_predictions(model, X_test)

    evaluation_result = calculate_metrics(y_test, test_predictions)

    comparison_table = create_comparison_table(y_test, test_predictions)

    predicted_price = float(model.predict(latest_row)[0])

    current_price = float(history["Close"].iloc[-1])

    return PredictionResult(
        ticker=validated_ticker,
        current_price=current_price,
        predicted_price=round(predicted_price, 2),
        price_change=round(predicted_price - current_price, 2),
        evaluation=evaluation_result,
        historical_prices=history["Close"],
        comparison_table=comparison_table,
    )


if __name__ == "__main__":
    try:
        result = predict_next_close("MSFT")

        print("Stock Price Prediction")
        print("-" * 40)
        print(f"Ticker              : {result.ticker}")
        print(f"Current Price       : ${result.current_price:.2f}")
        print(f"Predicted Next Close: ${result.predicted_price:.2f}")
        print(f"Predicted Change    : ${result.price_change:+.2f}")

    except ValueError as error:
        print(f"Prediction failed: {error}")
