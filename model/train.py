import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("../data/diabetes.csv")
X, y = data.drop("Outcome", axis=1), data["Outcome"]
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, "../model/diapredict_model.pkl")  # Save model
print("✅ Model trained & saved!")