import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the data
df = pd.read_csv("students.csv")

# Features (inputs) and target (what we want to predict)
X = df[["attendance", "avg_marks", "fee_pending"]]
y = df["dropout"]

# Split data into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "dropout_model.pkl")
print("Model trained and saved as dropout_model.pkl")
