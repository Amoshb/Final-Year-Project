
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import load_model
from bayes_opt import BayesianOptimization
from trading_singnal_analysis import main_trading

# ========== MODEL CONFIGURATION ==========
features_config = {
    "sentiment_score": ["sentiment_score"],
    "ema_macd": ["EMA_5", "EMA_8", "EMA_13", "EMA_12", "EMA_26", "MACD", "Signal_Line"],
    "fibonacci": ["Fib_23", "Fib_38", "Fib_50", "Fib_61", "Fib_78"],
    "stochastic": ["Stoch_K", "Stoch_D"],
    "fundamental": ["impact", "actual", "forecast", "previous", "eurusd_signal"],
    "obv": ["OBV"],
    "price": ["Open", "High", "Low"],
    "bollinger": ["BB_Mid", "BB_Upper", "BB_Lower"],
    "atr": ["ATR_14"],
    "rsi": ["RSI_14"]
}

models_config = {
    "fib_72.h5": features_config["price"] + features_config["fibonacci"],
    "all_technical_indicators.h5": features_config["price"] + features_config["ema_macd"] + features_config["fibonacci"] + features_config["stochastic"] + features_config["obv"] + features_config["bollinger"] + features_config["atr"],
    "ema_macd_fibo_82.h5": features_config["price"] + features_config["ema_macd"] + features_config["fibonacci"],
    "fib_fun_80.h5": features_config["price"] + features_config["fibonacci"] + features_config["fundamental"],
    "fib_senti_70.h5": features_config["price"] + features_config["fibonacci"] + features_config["sentiment_score"],
    "fun_sen_73.h5": features_config["price"] + features_config["fundamental"] + features_config["sentiment_score"],
    "funda_77.h5": features_config["price"] + features_config["fundamental"],
    "sentiment_79.h5": features_config["price"] + features_config["sentiment_score"]
}

# ========== FOREX DATA ==========
forex_df = pd.read_csv("final_merged_forex_data.csv")
forex_df["timestamp"] = pd.to_datetime(forex_df["Datetime"])

# ========== DATA SEQUENCE BUILDER ==========
def prepare_sequences(df, features, target="Close", seq_length=20):
    X = df[features].values
    y = df[[target]].values
    X_scaled = MinMaxScaler().fit_transform(X)
    y_scaled = MinMaxScaler().fit_transform(y)
    data = np.hstack((X_scaled, y_scaled))
    test_data = data[int(len(df)*0.9):]  # Use last 10% for testing

    X_seq, y_seq = [], []
    for i in range(len(test_data) - seq_length):
        X_seq.append(test_data[i:i+seq_length, :-1])
        y_seq.append(test_data[i+seq_length, -1])
    return np.array(X_seq), np.array(y_seq), MinMaxScaler().fit(y)

# ========== PREDICTIONS ==========
def predict_with_model(model_path, features):
    model = load_model(model_path)
    X_test, _, scaler_y = prepare_sequences(forex_df.copy(), features)
    if X_test.size == 0:
        raise ValueError("Empty test data after sequence creation.")
    predictions = model.predict(X_test)
    return scaler_y.inverse_transform(predictions).flatten()

# ========== LOAD AND EVALUATE MODELS ==========
predictions_dict = {}

for model_name, features in models_config.items():
    print(f"Loading model: {model_name}")
    try:
        path = os.path.join("with_test", model_name)
        predictions = predict_with_model(path, features)
        predictions_dict[model_name] = predictions
    except Exception as e:
        print(f"Error with {model_name}: {e}")

print("\n All predictions generated!")

# ========== EVALUATION ==========
actual_prices = forex_df["Close"].values[-len(next(iter(predictions_dict.values()))) :]
rolling_window = 300
rmse_values, model_scores = [], {}

for model_name, preds in predictions_dict.items():
    actual = actual_prices[-rolling_window:]
    pred = preds[-rolling_window:]
    r2 = max(0, r2_score(actual, pred))
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    score = (r2 * 0.4) + ((1 / (mae + 1e-5)) * 0.3) + ((1 / (rmse + 1e-5)) * 0.3)
    rmse_values.append(rmse)
    model_scores[model_name] = (r2, mae, rmse, score)

threshold = np.median(rmse_values) * 2
top_models = sorted(
    {k: v for k, v in model_scores.items() if v[2] < threshold}.items(),
    key=lambda x: x[1][3], reverse=True
)[:3]

best_predictions = [predictions_dict[m[0]] for m in top_models]

print("\nSelected Best Models:")
for m in top_models:
    print(m[0])

# ========== BAYESIAN OPTIMIZATION ==========
def objective_function(w1, w2, w3):
    weights = np.array([w1, w2, w3])
    weights /= weights.sum()
    combined = np.average(np.array(best_predictions), axis=0, weights=weights)
    return -mean_absolute_error(actual_prices, combined)

bo = BayesianOptimization(
    f=objective_function,
    pbounds={"w1": (0.1, 0.5), "w2": (0.1, 0.5), "w3": (0.1, 0.5)},
    random_state=42
)
bo.maximize(init_points=5, n_iter=20)
best_weights = np.array([bo.max['params']['w1'], bo.max['params']['w2'], bo.max['params']['w3']])
best_weights /= best_weights.sum()
ensemble_predictions = np.average(np.array(best_predictions), axis=0, weights=best_weights)

print("\n Optimized Weights:", best_weights)

# ========== FINAL EVALUATION ==========
ensemble_r2 = r2_score(actual_prices, ensemble_predictions)
ensemble_mape = mean_absolute_percentage_error(actual_prices, ensemble_predictions)
ensemble_mae = mean_absolute_error(actual_prices, ensemble_predictions)
ensemble_rmse = np.sqrt(mean_squared_error(actual_prices, ensemble_predictions))

results = pd.DataFrame({
    "Metric": ["R2 Score", "MAPE", "MAE", "RMSE"],
    "Value": [ensemble_r2, ensemble_mape, ensemble_mae, ensemble_rmse]
})

print("\n Ensemble Model Evaluation:")
print(results)

# ========== PLOT ==========
plt.figure(figsize=(14, 7))
plt.plot(actual_prices, label="Actual", color="blue")
plt.plot(ensemble_predictions, label="Predicted", color="red", linestyle="--")
plt.title("Actual vs Ensemble Predictions")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

analysis_df = pd.DataFrame({"Actual_Close": actual_prices, "Pred_Close": ensemble_predictions})
main_trading(analysis_df)