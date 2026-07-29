# 📈 Stock Price Prediction System

A complete beginner-friendly Machine Learning project built using **Python**, **Scikit-learn**, **yfinance**, and **Streamlit**.

The application predicts a stock's **Next Trading Day Closing Price** based on recent market data such as price history, trading volume, and simple technical indicators.

---

# Project Overview

This project demonstrates the complete Machine Learning development lifecycle.

- Problem Understanding
- Market Data Collection
- Feature Engineering
- Model Training
- Model Evaluation
- Model Serialization
- Prediction
- Streamlit Web Application
- Deployment

The project follows a clean, modular architecture suitable for beginners as well as professional software development.

---

# Project Objective

To predict a stock's expected closing price for the next trading day using a trained **Linear Regression** Machine Learning model.

The prediction is based on the following market features:

- Open, High, Low, Close prices
- Trading Volume
- 5-day and 20-day moving averages
- Daily return
- High-Low and Open-Close ranges

The model predicts:

- Next Trading Day Closing Price

---

# Technology Stack

| Technology | Purpose |
|------------|---------|
| Python 3.13+ | Programming Language |
| yfinance | Market Data Retrieval |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Scikit-learn | Machine Learning |
| Joblib | Model Serialization |
| Streamlit | Web Application |
| VS Code | Development Environment |
| Git & GitHub | Version Control |

---

# Project Structure

```
StockPricePrediction/
│
├── app/
│   └── app.py
│
├── assets/
│
├── data/
│
├── model/
│   └── <TICKER>_linear_regression_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Machine Learning Workflow

```
Problem Statement
        │
        ▼
Download Market Data (yfinance)
        │
        ▼
Feature Engineering
        │
        ▼
Target Variable (Next Close)
        │
        ▼
Chronological Train-Test Split
        │
        ▼
Train Linear Regression Model
        │
        ▼
Model Evaluation
        │
        ▼
Save Model (.pkl)
        │
        ▼
Streamlit Application
        │
        ▼
Prediction
        │
        ▼
Deployment
```

---

# Installation

## Step 1

Clone the repository.

```bash
git clone <repository-url>
```

---

## Step 2

Open the project.

```bash
cd StockPricePrediction
```

---

## Step 3

Create a virtual environment.

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Step 4

Install required packages.

```bash
pip install -r requirements.txt
```

---

# Market Data

Unlike a fixed CSV dataset, this project downloads live historical
data for any ticker directly from Yahoo Finance using **yfinance**.

No manual dataset setup is required. Data is retrieved automatically
when a ticker is submitted, based on the following configuration:

| Setting | Value |
|---------|-------|
| Default Ticker | MSFT |
| History Window | 5 years |
| Split Method | Chronological (80% train / 20% test) |

---

# Training the Model

Run:

```bash
python src/train_model.py
```

The trained model will be stored inside:

```
model/
```

Generated file:

```
<TICKER>_linear_regression_model.pkl
```

---

# Evaluating the Model

Run:

```bash
python src/evaluate.py
```

The following metrics will be displayed:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# Running the Streamlit Application

Execute:

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

# Application Features

- User-friendly interface
- Ticker validation
- Live market data retrieval
- Real-time prediction
- Historical price and evaluation charts
- Modular architecture
- Easy to maintain
- Beginner-friendly code

---

# Project Modules

## config.py

Stores project configuration.

---

## preprocess.py

Downloads market data and engineers model features.

---

## train_model.py

Trains the Linear Regression model.

---

## evaluate.py

Calculates model performance metrics.

---

## predict.py

Runs the full pipeline and predicts the next closing price.

---

## app.py

Provides the Streamlit web interface.

---

# Expected Input

| Feature | Example |
|----------|---------|
| Ticker | MSFT |

---

# Expected Output

```
Predicted Next Close

$421.36
```

---

# Learning Outcomes

After completing this project, you will be able to:

- Understand the Machine Learning lifecycle
- Retrieve and work with live market data
- Engineer technical indicator features
- Perform a chronological train-test split for time series data
- Train a Linear Regression model
- Evaluate model performance
- Save and load Machine Learning models
- Build interactive Streamlit applications
- Deploy Machine Learning projects
- Organize projects using clean architecture

---

# Future Enhancements

- Random Forest / XGBoost Regression
- LSTM / Time Series Deep Learning Models
- Hyperparameter Tuning
- Additional Technical Indicators (RSI, MACD, Bollinger Bands)
- Multi-Day Forecast Horizon
- Cross Validation
- Model Comparison
- Cloud Deployment
- Portfolio-Level Predictions

---

# Developed For

**Machine Learning with Python**

Beginner Project Series

Project 2

Stock Price Prediction System

---

# Disclaimer

This project is developed for **educational purposes only** and does
not constitute financial advice. Predictions are estimates based on
historical data and should not be used as the sole basis for
investment decisions.

---

# License

This project is developed for educational purposes.

Students are encouraged to modify, improve, and extend the project for learning and practice.
