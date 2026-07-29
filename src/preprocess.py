"""
Data preprocessing module for the Stock Price Prediction System.

Responsibilities:
- Download historical market data for a ticker
- Validate the ticker symbol and downloaded data
- Engineer technical indicator features
- Build the prediction target (tomorrow's closing price)
- Handle missing values created by rolling calculations
- Separate features and target variable

This file does not train or evaluate the Machine Learning model.
"""

import re

import pandas as pd
import yfinance as yf

from config import (
    DATA_PERIOD,
    FEATURE_COLUMNS,
    MAX_TICKER_LENGTH,
    MIN_REQUIRED_ROWS,
    TARGET_COLUMN,
)


TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,10}$")


def validate_ticker(ticker: str) -> str:
    """
    Validate and normalize a stock ticker symbol.

    Args:
        ticker: Ticker symbol entered by the user.

    Returns:
        The normalized (uppercase, stripped) ticker symbol.

    Raises:
        ValueError: If the ticker is empty, too long, or contains
            characters that are not valid in a ticker symbol.
    """

    if ticker is None:
        raise ValueError("A stock ticker must be provided.")

    normalized_ticker = ticker.strip().upper()

    if not normalized_ticker:
        raise ValueError("A stock ticker must be provided.")

    if len(normalized_ticker) > MAX_TICKER_LENGTH:
        raise ValueError(
            f"Ticker symbols must be {MAX_TICKER_LENGTH} characters or fewer."
        )

    if not TICKER_PATTERN.match(normalized_ticker):
        raise ValueError(
            f"'{ticker}' is not a valid ticker symbol."
        )

    return normalized_ticker


def download_market_data(
    ticker: str,
    period: str = DATA_PERIOD,
) -> pd.DataFrame:
    """
    Download historical market data for a ticker using yfinance.

    Args:
        ticker: Validated stock ticker symbol.
        period: Historical data window (e.g. "5y").

    Returns:
        A Pandas DataFrame containing OHLCV market data.

    Raises:
        ValueError: If no data is returned for the ticker.
    """

    dataframe = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"No market data was found for ticker '{ticker}'. "
            "Check that the symbol is correct."
        )

    # yfinance can return multi-level columns for some tickers.
    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = dataframe.columns.get_level_values(0)

    return dataframe


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Build technical indicator features from raw market data.

    Features created:
    - MA5: 5-day moving average of the closing price
    - MA20: 20-day moving average of the closing price
    - Return: Daily percentage change in closing price
    - High_Low: Difference between the day's high and low
    - Open_Close: Difference between the day's open and close
    - Target: Tomorrow's closing price (prediction target)

    Args:
        dataframe: Raw OHLCV market data.

    Returns:
        DataFrame with engineered features and the target column.
    """

    engineered_dataframe = dataframe.copy()

    engineered_dataframe["MA5"] = (
        engineered_dataframe["Close"].rolling(5).mean()
    )
    engineered_dataframe["MA20"] = (
        engineered_dataframe["Close"].rolling(20).mean()
    )

    engineered_dataframe["Return"] = (
        engineered_dataframe["Close"].pct_change()
    )

    engineered_dataframe["High_Low"] = (
        engineered_dataframe["High"] - engineered_dataframe["Low"]
    )
    engineered_dataframe["Open_Close"] = (
        engineered_dataframe["Open"] - engineered_dataframe["Close"]
    )

    engineered_dataframe[TARGET_COLUMN] = (
        engineered_dataframe["Close"].shift(-1)
    )

    return engineered_dataframe


def validate_row_count(dataframe: pd.DataFrame) -> None:
    """
    Validate that enough rows remain to train a usable model.

    Args:
        dataframe: Feature-engineered market data.

    Raises:
        ValueError: If fewer than the minimum required rows are available.
    """

    if len(dataframe) < MIN_REQUIRED_ROWS:
        raise ValueError(
            "Not enough historical data is available to train a "
            f"reliable model (found {len(dataframe)} usable rows, "
            f"need at least {MIN_REQUIRED_ROWS})."
        )


def preprocess_dataset(
    ticker: str,
    period: str = DATA_PERIOD,
) -> pd.DataFrame:
    """
    Execute the complete market data preprocessing workflow.

    Steps:
    1. Validate the ticker symbol
    2. Download historical market data
    3. Engineer technical indicator features
    4. Remove rows with missing values from rolling calculations
    5. Validate that enough rows remain for training

    Args:
        ticker: Stock ticker symbol.
        period: Historical data window (e.g. "5y").

    Returns:
        A clean DataFrame ready for model training, including the most
        recent row (used later to predict tomorrow's price).
    """

    validated_ticker = validate_ticker(ticker)

    raw_dataframe = download_market_data(validated_ticker, period)

    engineered_dataframe = engineer_features(raw_dataframe)

    # The most recent row has no target (tomorrow has not happened yet),
    # so it is kept separately for the final prediction step.
    latest_row = engineered_dataframe.iloc[[-1]][FEATURE_COLUMNS]

    training_dataframe = engineered_dataframe.dropna().copy()

    validate_row_count(training_dataframe)

    return training_dataframe, latest_row


def prepare_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate input features and target values.

    Args:
        dataframe: Preprocessed market data.

    Returns:
        A tuple containing:
        - X: Input feature DataFrame
        - y: Target Series containing tomorrow's closing price
    """

    features = dataframe[FEATURE_COLUMNS].copy()
    target = dataframe[TARGET_COLUMN].copy()

    return features, target


def get_preprocessed_data(
    ticker: str,
    period: str = DATA_PERIOD,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Download, clean, and separate the dataset for a ticker.

    This is the main function that other project modules should call.

    Args:
        ticker: Stock ticker symbol.
        period: Historical data window (e.g. "5y").

    Returns:
        A tuple containing:
        - features: Input feature DataFrame
        - target: Target Series (tomorrow's closing price)
        - full_history: Full cleaned historical DataFrame (for charts)
        - latest_row: Most recent row of features (for prediction)
    """

    training_dataframe, latest_row = preprocess_dataset(ticker, period)

    features, target = prepare_features_and_target(training_dataframe)

    return features, target, training_dataframe, latest_row


if __name__ == "__main__":
    try:
        clean_features, clean_target, history, latest = (
            get_preprocessed_data("MSFT")
        )

        print("Dataset preprocessing completed successfully.")
        print(f"Total records: {len(clean_features)}")
        print(f"Total columns: {len(clean_features.columns)}")
        print("\nFirst five records:")
        print(history.head())

    except ValueError as error:
        print(f"Preprocessing failed: {error}")
