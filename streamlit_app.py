"""
Streamlit application for the Stock Price Prediction System.

Responsibilities:
- Display the project interface
- Accept a stock ticker symbol from the user
- Call the prediction function
- Display the predicted next closing price
- Show historical and actual-versus-predicted charts
- Show beginner-friendly error messages

This file does not train or evaluate the Machine Learning model directly.
"""

import sys
from pathlib import Path

import streamlit as st


# Add the project root directory to Python's import path.
# This allows app.py to import modules from the src folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.predict import predict_next_close  # noqa: E402
from src.evaluate import interpret_r2_score  # noqa: E402


def configure_page() -> None:
    """
    Configure the Streamlit browser page.

    This function sets the title, icon, layout, and sidebar state.
    """

    st.set_page_config(
        page_title="Stock Price Prediction",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def display_header() -> None:
    """
    Display the application title and introductory information.
    """

    st.title("📈 Stock Price Prediction System")

    st.write(
        "Enter a stock ticker symbol to predict the expected "
        "closing price for the next trading day."
    )

    st.info(
        "This application uses a Linear Regression model trained on "
        "recent historical market data."
    )


def display_sidebar() -> None:
    """
    Display project information in the Streamlit sidebar.
    """

    with st.sidebar:
        st.header("Project Information")

        st.write(
            """
            **Project:** Stock Price Prediction

            **Algorithm:** Linear Regression

            **Framework:** Streamlit

            **Data Source:** Yahoo Finance (yfinance)
            """
        )

        st.divider()

        st.subheader("Input Features")

        st.write(
            """
            - Open, High, Low, Close, Volume
            - 5-day moving average (MA5)
            - 20-day moving average (MA20)
            - Daily return
            - High-Low range
            - Open-Close range
            """
        )

        st.divider()

        st.caption(
            "The prediction is an estimate generated from historical "
            "data and is not financial advice."
        )


def get_ticker_input() -> str:
    """
    Display the input control and collect the requested ticker symbol.

    Returns:
        The raw ticker symbol entered by the user.
    """

    st.subheader("Stock Ticker")

    return st.text_input(
        label="Enter Stock Ticker",
        value="MSFT",
        help="Enter a valid ticker symbol, e.g. MSFT, AAPL, GOOGL.",
    )


def display_prediction_result(result) -> None:
    """
    Display the prediction result, evaluation metrics, and charts.

    Args:
        result: PredictionResult returned by predict_next_close().
    """

    st.success("Prediction completed successfully.")

    st.subheader("Prediction")

    price_column, change_column = st.columns(2)

    with price_column:
        st.metric(
            label="Current Price",
            value=f"${result.current_price:.2f}",
        )

    with change_column:
        st.metric(
            label="Predicted Next Close",
            value=f"${result.predicted_price:.2f}",
            delta=f"{result.price_change:+.2f}",
        )

    st.subheader("Model Evaluation")

    mae_column, rmse_column, r2_column = st.columns(3)

    with mae_column:
        st.metric(
            "MAE",
            f"${result.evaluation.mean_absolute_error:.2f}",
        )

    with rmse_column:
        st.metric(
            "RMSE",
            f"${result.evaluation.root_mean_squared_error:.2f}",
        )

    with r2_column:
        st.metric(
            "R² Score",
            f"{result.evaluation.r2_score:.3f}",
        )

    st.caption(interpret_r2_score(result.evaluation.r2_score))

    st.divider()

    st.subheader("Historical Closing Price")
    st.line_chart(result.historical_prices)

    st.subheader("Actual vs Predicted (Test Period)")
    st.line_chart(result.comparison_table[["Actual", "Predicted"]])

    st.subheader("Recent Data")
    st.dataframe(result.comparison_table.tail())


def main() -> None:
    """
    Run the Stock Price Prediction Streamlit application.
    """

    configure_page()
    display_header()
    display_sidebar()

    with st.form("stock_prediction_form"):
        ticker_input = get_ticker_input()

        submit_button = st.form_submit_button(
            label="Predict",
            type="primary",
            use_container_width=True,
        )

    if submit_button:
        try:
            with st.spinner(f"Analyzing {ticker_input.upper()}..."):
                result = predict_next_close(ticker_input)

            display_prediction_result(result)

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                "An unexpected error occurred while generating the prediction."
            )

            st.exception(error)


if __name__ == "__main__":
    main()
