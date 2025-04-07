from datetime import timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

# Load new Forex data
forex_df = pd.read_csv("final_merged_forex_data.csv")
forex_df["timestamp"] = pd.to_datetime(forex_df["Datetime"])

# Features configuration
sentiment_features = ["sentiment_score"]
moving_average_features = ["EMA_5", "EMA_8", "EMA_13"]
macd_features = ["EMA_12","EMA_26", "MACD", "Signal_Line"]
fibonacci_features = [ "Fib_23", "Fib_38", "Fib_50", "Fib_61", "Fib_78"]
stochastic_features = ["Stoch_K", "Stoch_D"]
fundamental_features = ["impact", "actual", "forecast", "previous","eurusd_signal"]
obv_features = ["OBV"]
price_features = ["Open", "High", "Low"]
bollinger_band_features = ["BB_Mid", "BB_Upper", "BB_Lower"]
atr_features = ["ATR_14"]
rsi_features = ["RSI_14"]

models_config = {
    "fib_72.h5": price_features + fibonacci_features,
    "all_technical_indicators.h5":price_features + moving_average_features + macd_features + fibonacci_features + stochastic_features + obv_features + bollinger_band_features + atr_features ,
    "ema_macd_fibo_82.h5":price_features + moving_average_features + macd_features + fibonacci_features,
    "fib_fun_80.h5":price_features + fibonacci_features+ fundamental_features,
    #"fib_low_vali_high_test.h5":price_features + fibonacci_features,
    "fib_senti_70.h5":price_features + fibonacci_features + sentiment_features,
    "fun_sen_73.h5":price_features + fundamental_features + sentiment_features,
    "funda_77.h5":price_features + fundamental_features,
    "sentiment_79.h5": price_features + sentiment_features
}

# Static selection of models (Manually chosen best-performing models)
selected_models = [
    "sentiment_79.h5",
    "ema_macd_fibo_82.h5",
    "fun_sen_73.h5"
]

def multi(features, forex_df, model, target_column='Close'):
    # Normalize data using MinMaxScaler
    X = forex_df[features].values
    y = forex_df[[target_column]].values  

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)

    # Combine scaled features and target
    df_scaled = np.hstack((X_scaled, y_scaled))

    # Split dataset
    train_size = int(len(forex_df) * 0.75)
    validation_size = int(len(forex_df) * 0.15)
    test_data = df_scaled[train_size + validation_size:]

    # Function to create sequences
    def create_sequences(data, seq_length=20):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length, :-1])
            y.append(data[i+seq_length, -1])
        return np.array(X), np.array(y)

    seq_length = 20
    X_test, y_test = create_sequences(test_data, seq_length)

    # Ensure test dataset is not empty
    if X_test.size == 0:
        raise ValueError("Test dataset is empty. Check data preparation steps.")

    # Predict on test data
    test_predictions = model.predict(X_test)
    test_predictions_rescaled = scaler_y.inverse_transform(test_predictions).flatten()
    return test_predictions_rescaled

# Load models and perform predictions
predictions_dict = {}

for model_name in selected_models:
    print(f"\n🔹 Loading model: {model_name}")
    import os
    model_path = os.path.join("with_test", model_name)
    model = load_model(model_path)

    try:
        features = models_config[model_name]
        predictions = multi(features, forex_df.copy(), model)
        predictions_dict[model_name] = predictions
    except ValueError as e:
        print(f"❌ Skipping {model_name} due to error: {e}")
        continue

print("\n✅ All predictions generated!")

# Get actual prices for comparison
actual_prices = forex_df["Close"].values[-len(next(iter(predictions_dict.values()))):]

# Static ensemble learning: Equal weighting
ensemble_predictions = np.mean(np.array(list(predictions_dict.values())), axis=0)

# Evaluate the ensemble model
mae = mean_absolute_error(actual_prices, ensemble_predictions)
mse = mean_squared_error(actual_prices, ensemble_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actual_prices, ensemble_predictions)
ensemble_mape = mean_absolute_percentage_error(actual_prices, ensemble_predictions)

print("\n📊 Static Ensemble Model Evaluation:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Mape² Score: {ensemble_mape:.4f}")

# Plot predictions vs actual values
plt.figure(figsize=(12, 5))
plt.plot(actual_prices, label="Actual Prices", color="blue")
plt.plot(ensemble_predictions, label="Ensemble Prediction", color="red", linestyle="dashed")
plt.legend()
plt.title("Static Ensemble Learning: Actual vs Predicted Prices")
plt.show()
