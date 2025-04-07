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

# Define features for each model
sentiment_features = ["sentiment_score"]
moving_average_features = ["EMA_5", "EMA_8", "EMA_13"]
macd_features = ["EMA_12", "EMA_26", "MACD", "Signal_Line"]
fibonacci_features = ["Fib_23", "Fib_38", "Fib_50", "Fib_61", "Fib_78"]
stochastic_features = ["Stoch_K", "Stoch_D"]
fundamental_features = ["impact", "actual", "forecast", "previous", "eurusd_signal"]
obv_features = ["OBV"]
price_features = ["Open", "High", "Low"]
bollinger_band_features = ["BB_Mid", "BB_Upper", "BB_Lower"]
atr_features = ["ATR_14"]
rsi_features = ["RSI_14"]

# Define models and corresponding features
models_config = {
    "fib_72.h5": price_features + fibonacci_features,
    "all_technical_indicators.h5": price_features + moving_average_features + macd_features + fibonacci_features + stochastic_features + obv_features + bollinger_band_features + atr_features,
    "ema_macd_fibo_82.h5": price_features + moving_average_features + macd_features + fibonacci_features,
    "fib_fun_80.h5": price_features + fibonacci_features + fundamental_features,
    "fib_low_vali_high_test.h5": price_features + fibonacci_features,
    "fib_senti_70.h5": price_features + fibonacci_features + sentiment_features,
    "fun_sen_73.h5": price_features + fundamental_features + sentiment_features,
    "funda_77.h5": price_features + fundamental_features,
    "sentiment_79.h5": price_features + sentiment_features
}

# Specify model name to analyze
selected_model_name = "all_technical_indicators.h5"  # Change this to the specific model you want
features = models_config[selected_model_name]

def evaluate_model(features, forex_df, model, target_column='Close'):
    # Normalize data using MinMaxScaler
    X = forex_df[features].values  # Convert to NumPy array
    y = forex_df[[target_column]].values  # Keep as 2D array

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)

    # Combine scaled features and target
    df_scaled = np.hstack((X_scaled, y_scaled))

    # Split dataset
    train_size = int(len(forex_df) * 0.75)
    validation_size = int(len(forex_df) * 0.15)
    train_data = df_scaled[:train_size]
    validation_data = df_scaled[train_size:train_size + validation_size]
    test_data = df_scaled[train_size + validation_size:]

    # Function to create sequences
    def create_sequences(data, seq_length=20):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length, :-1])  # Features only
            y.append(data[i+seq_length, -1])  # Target (Close price)
        return np.array(X), np.array(y)

    seq_length = 20
    X_val, y_val = create_sequences(validation_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # Ensure datasets are not empty
    if X_val.size == 0 or X_test.size == 0:
        raise ValueError("Validation or test dataset is empty. Check data preparation steps.")

    # Predict on validation and test data
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    val_predictions_rescaled = scaler_y.inverse_transform(val_predictions).flatten()
    y_val_rescaled = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

    test_predictions_rescaled = scaler_y.inverse_transform(test_predictions).flatten()
    y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate evaluation metrics for validation
    val_mae = mean_absolute_error(y_val_rescaled, val_predictions_rescaled)
    val_mse = mean_squared_error(y_val_rescaled, val_predictions_rescaled)
    val_rmse = np.sqrt(val_mse)
    val_mape = mean_absolute_percentage_error(y_val_rescaled, val_predictions_rescaled)
    val_r2 = r2_score(y_val_rescaled, val_predictions_rescaled)

    # Calculate evaluation metrics for test
    test_mae = mean_absolute_error(y_test_rescaled, test_predictions_rescaled)
    test_mse = mean_squared_error(y_test_rescaled, test_predictions_rescaled)
    test_rmse = np.sqrt(test_mse)
    test_mape = mean_absolute_percentage_error(y_test_rescaled, test_predictions_rescaled)
    test_r2 = r2_score(y_test_rescaled, test_predictions_rescaled)

    print("\n🔹 Validation Evaluation Metrics:")
    print(f"MAE: {val_mae:.4f}")
    print(f"MSE: {val_mse:.4f}")
    print(f"RMSE: {val_rmse:.4f}")
    print(f"MAPE: {val_mape:.4f}")
    print(f"R2 Score: {val_r2:.4f}")

    print("\n🔹 Test Evaluation Metrics:")
    print(f"MAE: {test_mae:.4f}")
    print(f"MSE: {test_mse:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAPE: {test_mape:.4f}")
    print(f"R2 Score: {test_r2:.4f}")

    return val_predictions_rescaled, test_predictions_rescaled, y_val_rescaled, y_test_rescaled

# Load model
import os
model_path = os.path.join("with_test", selected_model_name)
model = load_model(model_path)  # Load trained model

# Generate predictions and evaluation
val_preds, test_preds, y_val_actual, y_test_actual = evaluate_model(features, forex_df.copy(), model)

# Define number of days to visualize
days_to_show = 120

# Get the last available timestamp in the dataset
last_date = forex_df["timestamp"].max()
start_date = last_date - timedelta(days=days_to_show)

# Filter only for plotting
plot_forex_df = forex_df[forex_df["timestamp"] >= start_date]

# Adjust time axis for plotting
time_axis = plot_forex_df["timestamp"].values[-len(test_preds):]

plt.figure(figsize=(14, 7))

# Plot actual prices for the filtered period
plt.plot(time_axis, plot_forex_df["Close"].values[-len(time_axis):], label="Actual Prices", color='black', linewidth=2)

# Plot test predictions for the filtered period
plt.plot(time_axis, test_preds[-len(time_axis):], linestyle='dashed', linewidth=1.5, color="red", label=f"Predicted")

plt.xlabel("Time", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.title(f"Actual vs Predicted Prices (Comprehensive Technical Analysis model)", fontsize=16)
plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\n Single-model evaluation complete.")
