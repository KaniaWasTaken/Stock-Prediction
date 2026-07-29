"""
Central configuration for the Stock Price Prediction project.

This module contains:
- Project directory paths
- Model file path pattern
- Ticker, data period, and feature settings
- Model training configuration

No data fetching, model training, or prediction logic should be
written in this file.
"""

from pathlib import Path
from typing import Final


# =========================================================
# PROJECT DIRECTORIES
# =========================================================

# Absolute path of the main project directory.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Folder where downloaded/cached market data may be stored.
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"

# Folder where trained Machine Learning models will be stored.
MODEL_DIR: Final[Path] = PROJECT_ROOT / "model"

# Folder containing the Streamlit application.
APP_DIR: Final[Path] = PROJECT_ROOT / "app"

# Folder containing optional images and other static resources.
ASSETS_DIR: Final[Path] = PROJECT_ROOT / "assets"


# =========================================================
# MODEL FILE PATH
# =========================================================

def get_model_file_path(ticker: str) -> Path:
    """
    Build the file path used to store the trained model for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g. "MSFT").

    Returns:
        Path where the trained Linear Regression model is saved.
    """

    safe_ticker = ticker.strip().upper()

    return MODEL_DIR / f"{safe_ticker}_linear_regression_model.pkl"


# =========================================================
# MARKET DATA CONFIGURATION
# =========================================================

# Ticker used by default when running modules directly from the CLI.
DEFAULT_TICKER: Final[str] = "MSFT"

# Historical data window downloaded for each ticker.
DATA_PERIOD: Final[str] = "5y"


# =========================================================
# DATASET CONFIGURATION
# =========================================================

# Input columns used by the model to predict the next closing price.
FEATURE_COLUMNS: Final[list[str]] = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA5",
    "MA20",
    "Return",
    "High_Low",
    "Open_Close",
]

# Output column that the Machine Learning model will predict.
TARGET_COLUMN: Final[str] = "Target"


# =========================================================
# MODEL TRAINING CONFIGURATION
# =========================================================

# 20% of the dataset (most recent rows) is held out for testing.
# The split is chronological, not random, since stock data is a
# time series and must not be shuffled.
TEST_SIZE: Final[float] = 0.20


# =========================================================
# VALID INPUT RANGES
# =========================================================

# Minimum number of rows required to train a usable model.
MIN_REQUIRED_ROWS: Final[int] = 60

# Maximum length accepted for a ticker symbol.
MAX_TICKER_LENGTH: Final[int] = 10


def create_required_directories() -> None:
    """
    Create the required project directories if they do not already exist.

    This function is useful when the project is executed for the first
    time and the model or assets folders have not yet been created.
    """

    directories = [
        DATA_DIR,
        MODEL_DIR,
        APP_DIR,
        ASSETS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
