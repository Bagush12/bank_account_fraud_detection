import pandas as pd
import xgboost as xgb

# Load your data
data = pd.read_csv("Variant I.csv")

# Separate features (X) and target (y)
X = data.drop("fraud_bool", axis=1)
y = data["fraud_bool"]

# Encode categorical columns using one-hot encoding
X_encoded = pd.get_dummies(X)

# Split your data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train your XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Predict on the test data
y_pred = xgb_model.predict(X_test)

# Evaluate your model and calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the model to a file (e.g., using joblib)
import joblib
joblib.dump(xgb_model, "xgb_model.pkl")