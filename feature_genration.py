import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# CONFIGURATION
ECONOMIC_FILE = "economic_calendar_data.csv"
NEWS_FILE = "dailyforex_eurusd_news.csv"
FOREX_FILE = "Cleaned_EURUSD60.csv"
MODEL_NAME = "yiyanghkust/finbert-tone"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMPACT_MAPPING = {
    "Low Volatility Expected": 1,
    "Moderate Volatility Expected": 2,
    "High Volatility Expected": 3
}
FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
POSITIVE_INDICATORS = ["GDP", "Retail Sales", "Industrial Production", "Manufacturing PMI", "Services PMI", "Employment", "Nonfarm Payrolls", "Trade Balance"]
INVERSE_INDICATORS = ["Unemployment Rate", "Jobless Claims", "Inflation Rate", "CPI", "PPI", "Government Debt"]

# DATA LOADING 
def load_csv(file_path):
    return pd.read_csv(file_path)

# VALUE CONVERSION
def convert_to_number(value):
    if not isinstance(value, str):
        return value
    value = value.strip().lower().replace(',', '')
    try:
        if value.endswith('%'):
            return float(value[:-1]) / 100
        elif 't' in value:
            return float(value.replace('t', '')) * 1e12
        elif 'b' in value:
            return float(value.replace('b', '')) * 1e9
        elif 'm' in value:
            return float(value.replace('m', '')) * 1e6
        elif 'k' in value:
            return float(value.replace('k', '')) * 1e3
        elif value in ["unknown", "unnown", "nan", "none", "", "-", "n/a"]:
            return 0
        return float(value)
    except:
        return 0

# ECONOMIC CLASSIFICATION
def classify_event(row):
    event = row["event"].lower()
    actual, forecast, country = row["actual"], row["forecast"], row["country"]
    is_pos = any(ind.lower() in event for ind in POSITIVE_INDICATORS)
    is_inv = any(ind.lower() in event for ind in INVERSE_INDICATORS)
    
    if country == "EUR":
        if is_pos:
            return 1 if actual > forecast else -1
        if is_inv:
            return 1 if actual < forecast else -1
    elif country == "USD":
        if is_pos:
            return -1 if actual > forecast else 1
        if is_inv:
            return -1 if actual < forecast else 1
    return 0

# SENTIMENT ANALYSIS 
def apply_sentiment_model(texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1, return_all_scores=True)

    def score_fn(text):
        result = classifier(text)[0]
        return round(result[2]["score"] - result[0]["score"], 4)

    tqdm.pandas()
    return texts.progress_apply(score_fn)

# TECHNICAL INDICATORS
def compute_indicators(df):
    for p in [5, 8, 13, 12, 26]:
        df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()

    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['BB_Mid'] + df['Close'].rolling(20).std() * 2
    df['BB_Lower'] = df['BB_Mid'] - df['Close'].rolling(20).std() * 2
    df['Lowest_Low'] = df['Low'].rolling(14).min()
    df['Highest_High'] = df['High'].rolling(14).max()
    df['Stoch_K'] = (df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low']) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    df['ATR_14'] = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    return rsi(df)

def rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    return df

# FIBONACCI 
def find_swing_highs_lows(df, lookback=50):
    df['Swing_High'] = df['High'].rolling(lookback).max()
    df['Swing_Low'] = df['Low'].rolling(lookback).min()
    highs, lows = [], []
    for i in range(lookback, len(df) - lookback):
        if df['High'].iloc[i] == df['Swing_High'].iloc[i]:
            highs.append(df.iloc[i])
        elif df['Low'].iloc[i] == df['Swing_Low'].iloc[i]:
            lows.append(df.iloc[i])
    return pd.DataFrame(highs), pd.DataFrame(lows)

def add_fibonacci_levels(df, highs, lows):
    df = df.copy()
    for level in FIB_LEVELS:
        df[f'Fib_{int(level * 100)}'] = np.nan
    for i in range(len(highs)-1):
        high = highs.iloc[i]['High']
        future_lows = lows[lows.index > highs.index[i]]
        if not future_lows.empty:
            low = future_lows.iloc[0]['Low']
            for level in FIB_LEVELS:
                df.loc[highs.index[i], f'Fib_{int(level * 100)}'] = high - (high - low) * level
    df.fillna(method='ffill', inplace=True)
    return df

# MAIN PIPELINE
def main():
    # Load & filter economic data
    eco_df = load_csv(ECONOMIC_FILE)
    eco_df = eco_df[eco_df["country"].isin(["USD", "EUR"])]
    for col in ['actual', 'forecast', 'previous']:
        eco_df[col] = eco_df[col].astype(str).apply(convert_to_number)
    eco_df["eurusd_signal"] = eco_df.apply(classify_event, axis=1)

    # Load and process news data
    news_df = load_csv(NEWS_FILE).dropna(subset=["description"])
    news_df["sentiment_score"] = apply_sentiment_model(news_df["description"].astype(str))

    # Load forex data
    forex_df = load_csv(FOREX_FILE)

    # Time conversion
    forex_df["timestamp"] = pd.to_datetime(forex_df["Datetime"])
    news_df["timestamp"] = pd.to_datetime(news_df["date"], format="%d/%m/%Y, %H:%M GMT0")
    eco_df["timestamp"] = pd.to_datetime(eco_df["date"], format="%Y/%m/%d %H:%M:%S")

    # Sort and merge
    forex_df.sort_values("timestamp", inplace=True)
    news_df.sort_values("timestamp", inplace=True)
    eco_df.sort_values("timestamp", inplace=True)

    merged = pd.merge_asof(forex_df, news_df[["timestamp", "sentiment_score"]], on="timestamp", direction="backward")
    merged = compute_indicators(merged)
    swing_highs, swing_lows = find_swing_highs_lows(merged)
    merged = add_fibonacci_levels(merged, swing_highs, swing_lows)
    merged = pd.merge_asof(merged, eco_df[["timestamp", "impact", "actual", "forecast", "previous", "eurusd_signal"]], on="timestamp", direction="backward")
    merged.fillna({"impact": "No Event", "actual": 0, "forecast": 0, "previous": 0}, inplace=True)
    merged["impact"] = merged["impact"].map(IMPACT_MAPPING).fillna(0)
    merged.drop(columns=["timestamp", "Swing_High", "Swing_Low"], inplace=True)
    merged.dropna(inplace=True)
    merged.to_csv("test_final_merged_forex_data.csv", index=False)
    print(merged)

if __name__ == "__main__":
    main()
