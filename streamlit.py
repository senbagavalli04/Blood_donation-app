import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load dataset
data = pd.read_csv("transfusion.csv")

# Check for null values
print(data.isnull().sum())

# Initialize a single MinMaxScaler for all features
scaler = MinMaxScaler()

# Scale all relevant columns using the same scaler
data[["Recency (months)", "Frequency (times)", "Monetary (c.c. blood)", "Time (months)"]] = scaler.fit_transform(
    data[["Recency (months)", "Frequency (times)", "Monetary (c.c. blood)", "Time (months)"]])

# Separate features and target
x = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target (last column)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train the RandomForestClassifier model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Save the trained RandomForest model
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the fitted scaler
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully.")
