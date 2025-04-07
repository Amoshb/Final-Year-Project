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
│── with_test                         # all trained models 
├── collect_ohlcv_data.py             # Collect historical OHLCV data from MetaTrader 5
├── collect_fundamental_data.py       # Scrape macroeconomic data from Investing.com
├── collect_news_sentiment_data.py    # Scrape financial news data for sentiment analysis
├── feature_genration.py              # Perform feature engineering and data integration
├── lstm.py                           # Train multiple LSTM forecasting models
├── ensemble_forex_trading.py         # Implement dynamic ensemble forecasting
├── trading_signal_analysis.py        # Generate trading signals and evaluate performance
│
├── base_line_lstm.py                 # Baseline LSTM model for comparison
├── svm_test.py                       # Support Vector Machine baseline model
├── xgboost_test.py                   # XGBoost baseline model
├── static_learning.py                # Static ensemble learning (manual weighting)
├── model_eval_test.py                # Optional: Evaluate trained models for accuracy
│
├── requirements.txt                  # Python package requirements
├── README.md                         # Project documentation (this file)
