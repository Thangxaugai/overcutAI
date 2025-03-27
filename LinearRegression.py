import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('data.csv')


feature_cols = [col for col in df.columns if col not in ['No', 'Overcut']]
print(f"\nUsing features: {feature_cols}")

# Prepare data for training
X = df[feature_cols]  # Use identified feature columns
y = df['Overcut']     # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")


model = LinearRegression()

# Train the model
print("\nTraining model...")
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.6f}")
print(f"R²: {r2:.6f}")
print(f"Intercept: {model.intercept_:.6f}")
print("\nFeature coefficients:")
for feature, coef in zip(feature_cols, model.coef_):
    print(f"- {feature}: {coef:.6f}")

# Load test data from test.csv
print("\n============================================")
print("Predicting values from test.csv")
print("============================================")
test_data = pd.read_csv('test.csv')
print("\nTest data:")
print(test_data)

# Check if we have the same features
test_X = test_data[feature_cols]
print("\nFeatures for prediction:")
print(test_X)

# Make prediction
predicted_overcut = model.predict(test_X)
print("\nPrediction Results:")
print(f"Predicted Overcut value: {predicted_overcut[0]:.6f}")

# Add prediction to test data for display
test_data['Predicted_Overcut'] = predicted_overcut
print("\nTest data with prediction:")
print(test_data[['No', 'I', 'Ton', 'Toff', 'Wire Feed', 'MRR', 'SR', 'Overcut', 'Predicted_Overcut']])