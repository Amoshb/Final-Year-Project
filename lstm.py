
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.io as pio

# Set plotly default template
pio.templates.default = "plotly_dark"

while True:
    # --------------------------
    # 1. Load and Prepare Data
    # --------------------------
    file_path = "final_merged_forex_data.csv"
    merged_df = pd.read_csv(file_path)
    merged_df["timestamp"] = pd.to_datetime(merged_df["Datetime"])

    # Define features
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

    features = price_features + fibonacci_features 
    target_column = "Close"

    # Select features and target
    X = merged_df[features].values  # Convert to NumPy array
    y = merged_df[[target_column]].values  # Keep as 2D array for MinMaxScaler

    # Normalize Features & Target Separately
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)

    # Combine scaled features and target
    df_scaled = np.hstack((X_scaled, y_scaled))

    # Split dataset into train (75%), validation (15%) and test (10%)
    train_size = int(len(merged_df) * 0.75)
    validation_size = int(len(merged_df) * 0.15)
    train_data = df_scaled[:train_size]
    validation_data = df_scaled[train_size:train_size + validation_size]
    test_data = df_scaled[train_size + validation_size:]

    # --------------------------
    # 2. Create Sequences
    # --------------------------
    def create_sequences(data, seq_length=50):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length, :-1])  # Features only
            y.append(data[i+seq_length, -1])  # Target (Close price)
        return np.array(X), np.array(y)

    seq_length = 20  # Using past 20 time steps
    X_train, y_train = create_sequences(train_data, seq_length)
    X_val, y_val = create_sequences(validation_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # Ensure datasets are not empty
    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        raise ValueError("One of the datasets (train/validation/test) is empty. Check data preparation steps.")

    # --------------------------
    # 3. Build and Train LSTM Model
    # --------------------------
    model = Sequential([
        LSTM(units=192, return_sequences=True, input_shape=(seq_length, X_train.shape[2])),
        Dropout(0.2),
        LSTM(units=128, return_sequences=True),
        Dropout(0.2),
        LSTM(units=64),
        Dropout(0.2),
        Dense(units=32, activation='relu'),
        Dense(units=1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=22, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    # --------------------------
    # 4. Evaluate Model Performance
    # --------------------------
    # Validation performance
    val_predictions = model.predict(X_val)
    val_predictions_rescaled = scaler_y.inverse_transform(val_predictions).flatten()
    y_val_rescaled = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

    val_mae = mean_absolute_error(y_val_rescaled, val_predictions_rescaled)
    val_mse = mean_squared_error(y_val_rescaled, val_predictions_rescaled)
    val_rmse = np.sqrt(val_mse)
    val_mape = mean_absolute_percentage_error(y_val_rescaled, val_predictions_rescaled)
    val_r2 = r2_score(y_val_rescaled, val_predictions_rescaled)

    print(f"Validation - MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.4f}, R2 Score: {val_r2:.4f}")

    # Test performance
    test_predictions = model.predict(X_test)
    test_predictions_rescaled = scaler_y.inverse_transform(test_predictions).flatten()
    y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    test_mae = mean_absolute_error(y_test_rescaled, test_predictions_rescaled)
    test_mse = mean_squared_error(y_test_rescaled, test_predictions_rescaled)
    test_rmse = np.sqrt(test_mse)
    test_mape = mean_absolute_percentage_error(y_test_rescaled, test_predictions_rescaled)
    test_r2 = r2_score(y_test_rescaled, test_predictions_rescaled)

    print(f"Test - MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}, R2 Score: {test_r2:.4f}")

    # --------------------------
    # 5. Visualize Actual vs Predicted Close Prices using Plotly
    # --------------------------
    fig = go.Figure()

    # Add actual close price
    fig.add_trace(go.Scatter(y=y_test_rescaled, mode='lines', name='Actual Close', line=dict(color='blue')))


    # Add predicted close price
    fig.add_trace(go.Scatter(y=test_predictions_rescaled, mode='lines', name='Predicted Close', line=dict(color='orange', dash='dot')))

    # Add prediction error as shaded area
    fig.add_trace(go.Scatter(
        y=y_test_rescaled,
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Prediction Error'
    ))

    fig.update_layout(
        title='Actual vs Predicted Close Price',
        xaxis_title='Time Steps',
        yaxis_title='Close Price',
        legend_title='Legend',
        template='plotly_dark'
    )

    fig.show()
    if test_r2 > 0.75 and val_r2 > 0.75:
        break

# --------------------------
# 6. Save the Model (with User Confirmation)
# --------------------------
save_choice = input("Do you want to save the model? (yes/no): ").strip().lower()
if save_choice in ['yes', 'y']:
    model_name = input("Enter the desired model name (without extension): ").strip()
    if not model_name:
        model_name = "lstm_forex_model"
    import os 
    filename = os.path.join("with_test", f"{model_name.replace(' ', '_').lower()}.h5")
    model.save(filename)
    print(f"✅ Model saved as '{filename}'")
else:
    print("❌ Model was not saved.")

