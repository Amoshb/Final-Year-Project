import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score

# --------------------------
# 1. Load and Prepare Data
# --------------------------
file_path = "final_merged_forex_data.csv"
merged_df = pd.read_csv(file_path)
merged_df["timestamp"] = pd.to_datetime(merged_df["Datetime"])

# Define Features (Ensure No Future Data is Used)
sentiment_features = ["sentiment_score"]
fibonacci_features = ["Fib_23", "Fib_38", "Fib_50", "Fib_61", "Fib_78"]
price_features = ["Open", "High", "Low"]

features = price_features + fibonacci_features
target_column = "Close"

# Select Features and Target
X = merged_df[features].values
y = merged_df[[target_column]].values  # Keep as 2D array for MinMaxScaler

# --------------------------
# 2. Split Dataset into Train (75%), Validation (15%), and Test (10%)
# --------------------------
total_samples = len(X)
train_size = int(total_samples * 0.75)
val_size = int(total_samples * 0.15)
test_size = total_samples - train_size - val_size  # Remaining 10%

# --------------------------
# 3. Proper Scaling (No Leakage)
# --------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit ONLY on training data
scaler_X.fit(X[:train_size])
scaler_y.fit(y[:train_size])

# Transform all datasets
X_train = scaler_X.transform(X[:train_size])
X_val = scaler_X.transform(X[train_size:train_size + val_size])
X_test = scaler_X.transform(X[train_size + val_size:])

y_train = scaler_y.transform(y[:train_size]).flatten()
y_val = scaler_y.transform(y[train_size:train_size + val_size]).flatten()
y_test = scaler_y.transform(y[train_size + val_size:]).flatten()

# --------------------------
# 4. Train XGBoost Model
# --------------------------
xgb_model = XGBRegressor(
    n_estimators=200,  # Number of trees
    learning_rate=0.05,  # Step size
    max_depth=6,  # Tree depth
    subsample=0.8,  # Fraction of samples per tree
    colsample_bytree=0.8,  # Fraction of features per tree
    objective='reg:squarederror',  # Loss function
    random_state=42
)

xgb_model.fit(X_train, y_train)



# --------------------------
# 5. Evaluate Model on Validation Set
# --------------------------
val_predictions = xgb_model.predict(X_val)

# Rescale predictions & actual values back to original scale
y_val_rescaled = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
val_predictions_rescaled = scaler_y.inverse_transform(val_predictions.reshape(-1, 1)).flatten()

# Compute Evaluation Metrics
val_mae = mean_absolute_error(y_val_rescaled, val_predictions_rescaled)
val_mse = mean_squared_error(y_val_rescaled, val_predictions_rescaled)
val_rmse = np.sqrt(val_mse)
val_mape = mean_absolute_percentage_error(y_val_rescaled, val_predictions_rescaled)
val_r2 = r2_score(y_val_rescaled, val_predictions_rescaled)

print(f"\n **Validation Set Metrics:**")
print(f"- MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}, MAPE: {val_mape:.2f}%\n")

# --------------------------
# 6. Evaluate Model on Test Set (Final Evaluation)
# --------------------------
test_predictions = xgb_model.predict(X_test)

# Rescale predictions & actual values back to original scale
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
test_predictions_rescaled = scaler_y.inverse_transform(test_predictions.reshape(-1, 1)).flatten()

# Compute Evaluation Metrics
test_mae = mean_absolute_error(y_test_rescaled, test_predictions_rescaled)
test_mse = mean_squared_error(y_test_rescaled, test_predictions_rescaled)
test_rmse = np.sqrt(test_mse)
test_mape = mean_absolute_percentage_error(y_test_rescaled, test_predictions_rescaled)
test_r2 = r2_score(y_test_rescaled, test_predictions_rescaled)

print(f"\n **Test Set Metrics:**")
print(f"- MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}, MAPE: {test_mape:.2f}%\n")

# # --------------------------
# # 8. Visualize Actual vs Predicted Close Prices (Test Set)
# # --------------------------
plt.figure(figsize=(14, 7))
plt.plot(y_test_rescaled, label='Actual Close', linewidth=2, color='blue')
plt.plot(test_predictions_rescaled, label='Predicted Close', linestyle='--', linewidth=2, color='orange')
plt.fill_between(range(len(y_test_rescaled)), y_test_rescaled, test_predictions_rescaled, color='gray', alpha=0.3, label='Prediction Error')
plt.title('Actual vs Predicted Close Price (XGBoost) - Test Set', fontsize=16)
plt.xlabel('Time Steps', fontsize=12)
plt.ylabel('Close Price', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# --------------------------
# 9. Visualize All Predictions in One Graph
# --------------------------

plt.figure(figsize=(14, 7))

# Train Set Predictions
train_predictions = xgb_model.predict(X_train)
train_predictions_rescaled = scaler_y.inverse_transform(train_predictions.reshape(-1, 1)).flatten()
y_train_rescaled = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()

# Validation Set Predictions
val_predictions = xgb_model.predict(X_val)
val_predictions_rescaled = scaler_y.inverse_transform(val_predictions.reshape(-1, 1)).flatten()
y_val_rescaled = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

# Test Set Predictions
test_predictions = xgb_model.predict(X_test)
test_predictions_rescaled = scaler_y.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Define Indexes for Separation
train_index = range(len(y_train_rescaled))
val_index = range(len(y_train_rescaled), len(y_train_rescaled) + len(y_val_rescaled))
test_index = range(len(y_train_rescaled) + len(y_val_rescaled), len(y_train_rescaled) + len(y_val_rescaled) + len(y_test_rescaled))

# Plot Actual vs Predicted for Train, Validation, and Test Sets
plt.plot(train_index, y_train_rescaled, label='Actual Train', color='blue', linewidth=2)
plt.plot(train_index, train_predictions_rescaled, linestyle='dashed', color='black', alpha=0.7, label='Predicted Train')

plt.plot(val_index, y_val_rescaled, label='Actual Validation', color='green', linewidth=2)
plt.plot(val_index, val_predictions_rescaled, linestyle='dashed', color='black', alpha=0.7, label='Predicted Validation')

plt.plot(test_index, y_test_rescaled, label='Actual Test', color='red', linewidth=2)
plt.plot(test_index, test_predictions_rescaled, linestyle='dashed', color='black', alpha=0.7, label='Predicted Test')

# Labels and Formatting
plt.title('XGBoost Actual vs Predicted Close Prices (Train, Validation, Test)', fontsize=16)
plt.xlabel('Time Steps', fontsize=12)
plt.ylabel('Close Price', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Show Plot
plt.show()
