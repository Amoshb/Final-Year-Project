
# EUR/USD Forex Forecasting System  
**Student ID:** 001183434  
**Final Year Project – University Submission**

## Project Overview

This project delivers an end-to-end forex forecasting and trading signal generation system for the EUR/USD currency pair. It integrates multiple data sources, including historical market data, macroeconomic indicators, and news sentiment analysis. Using a modular Python architecture, the system applies advanced feature engineering, deep learning (LSTM models), and ensemble learning techniques to generate reliable forex price forecasts and actionable trading signals.

Key project components include:
- Multi-source data collection (OHLCV data, fundamental events, financial news sentiment)
- Advanced feature engineering and data preprocessing
- Multiple LSTM model training
- Ensemble learning with Bayesian optimisation
- Trading signal generation and backtesting
- Evaluation through statistical and financial performance metrics

**Important:**  
> The complete source code has been submitted separately as required via Moodle.  
> For script descriptions and structure, refer to Appendix B in the final report.

---

## Project Structure

```bash
project/
│── with_test/                         # Folder containing all trained LSTM models (.h5 files)
│
├── collect_ohlcv_data.py              # Collect historical OHLCV data from MetaTrader 5
├── collect_fundamental_data.py        # Scrape macroeconomic data from Investing.com
├── collect_news_sentiment_data.py     # Scrape financial news data for sentiment analysis
├── feature_generation.py              # Perform feature engineering and data integration
├── lstm.py                            # Train multiple LSTM forecasting models
├── ensemble_forex_trading.py          # Implement dynamic ensemble forecasting
├── trading_signal_analysis.py         # Generate trading signals and evaluate performance
│
├── base_line_lstm.py                  # Baseline LSTM model for comparison
├── svm_test.py                        # Support Vector Machine baseline model
├── xgboost_test.py                    # XGBoost baseline model
├── static_learning.py                 # Static ensemble learning (manual weighting)
├── model_eval_test.py                 # Optional: Evaluate pre-trained models
│
├── Cleaned_EURUSD60.csv               # Raw historical OHLCV data
├── dailyforex_eurusd_news.csv         # Collected financial news sentiment data
├── economic_calendar_data.csv         # Collected macroeconomic events data
├── final_merged_forex_data.csv        # Final merged and processed dataset
│
├── requirements.txt                   # Python package dependencies
├── README.md                          # Project documentation (this file)
```

---

## Usage Instructions

### 1. Install Dependencies

First, ensure you have all the required Python packages installed:

```bash
pip install -r requirements.txt
```

> **Note:** If you have a compatible GPU, it is recommended to install TensorFlow with GPU support for faster model training.
> MetaTrader 5 terminal must be installed and configured properly.
> Also, make sure to enter your MT5 account credentials in the `collect_ohlcv_data.py` script:
>
> ```python
> # Add your login credentials
> login = 12345678
> password = 'yourpassword'
> server = 'YourBroker-Server'
> ```

### 2. Prepare the Environment

Ensure that the following CSV files are available in the project root (or run data collection scripts if not):

- `Cleaned_EURUSD60.csv` — Historical OHLCV data
- `economic_calendar_data.csv` — Macroeconomic events data
- `dailyforex_eurusd_news.csv` — Financial news sentiment data

> 💡 **Skip if already available:**  
> If all required data and pre-trained models are already present in the `with_test/` folder and data files are available, you can skip the data collection and model training phases.  
> You can directly proceed to running the ensemble and signal generation by executing:
>
> ```bash
> python ensemble_forex_trading.py
> ```

### 3. Data Collection (Optional if data is already prepared)

Run the following scripts to collect and prepare data:
```bash
python collect_ohlcv_data.py
python collect_fundamental_data.py
python collect_news_sentiment_data.py
```

### 4. Feature Engineering

After collecting data, process and generate features:
```bash
python feature_generation.py
```

### 5. Model Training (Optional if models are already trained)

Train the forecasting models:
```bash
python lstm.py
```

### 6. Ensemble Forecasting and Trading Signal Generation

Run the ensemble script to aggregate model predictions and generate trading signals:
```bash
python ensemble_forex_trading.py
```

### 7. (Optional) Baseline Model Evaluation

To compare baseline models:
```bash
python base_line_lstm.py
python svm_test.py
python xgboost_test.py
python static_learning.py
```

### 8. (Optional) Evaluate Pre-Trained Models

Evaluate existing trained models:
```bash
python model_eval_test.py
```

---

## 🔧 Customisation: Essential Code References for Running from Scratch

When running the project from scratch or changing the dataset, make sure to review and update the following code sections:

### 1. Update MetaTrader 5 Data Collection (`collect_ohlcv_data.py`)
Set your desired currency pair, timeframe, and date range:
```python
symbol = "EURUSD_cl"
timeframe = mt5.TIMEFRAME_H1
start_date = datetime(2024, 8, 23)
end_date = datetime(2025, 1, 23)
```

### 2. Update Economic Events Data Collection (`collect_fundamental_data.py`)
Modify the scraping date range:
```python
start_input.send_keys("08/23/2024")
end_input.send_keys("01/23/2025")
```

### 3. Update News Sentiment Data Collection (`collect_news_sentiment_data.py`)
Adjust the URL and page counter if needed:
```python
base_url = "https://www.dailyforex.com/currencies/eur/usd"
page_count = 1
max_pages = 30 # Adjust depending on your desired coverage
```

### 4. Update Feature Selection (`feature_generation.py`)
Define which features to include in your model. Example setup:
```python
features = price_features + fibonacci_features
target_column = "Close"
```
> 💡 **Important:**  
> When customising your dataset and selecting features in `feature_generation.py`, make sure to **keep track of the exact order of your features**.  
> This order **must match** the corresponding configuration inside `ensemble_forex_trading.py` under the `models_config` dictionary.  
> 
> Any mismatch between feature generation and model loading order will result in incorrect input for the trained models and potentially unreliable predictions.
> 
> ✅ **Tip:**  
> It is recommended to write down or comment your chosen feature order while editing `feature_generation.py`, and double-check it against your model configurations before training or running the ensemble.

### 5. Update Model Configurations (`ensemble_forex_trading.py`)
Ensure the model feature configurations match your selected features:
```python
models_config = {
    "fib_72.h5": features_config["price"] + features_config["fibonacci"],
    "all_technical_indicators.h5": features_config["price"] + features_config["ema_macd"] + features_config["fibonacci"] + features_config["stochastic"] + features_config["obv"] + features_config["bollinger"] + features_config["atr"],
    "ema_macd_fibo_82.h5": features_config["price"] + features_config["ema_macd"] + features_config["fibonacci"],
    ...
}
```

### 6. Update File Paths in Feature Generation (`feature_generation.py`)
Ensure file paths match your renamed data files:
```python
ECONOMIC_FILE = "economic_calendar_data.csv"
NEWS_FILE = "dailyforex_eurusd_news.csv"
FOREX_FILE = "Cleaned_EURUSD60.csv"
```

---

## Notes

- This project is designed for academic research purposes only and is not intended for live trading.
- Ensure API keys and credentials are securely managed.
- Respect external API terms of service when scraping or using data.