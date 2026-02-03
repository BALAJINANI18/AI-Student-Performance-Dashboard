import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Loading CSV...")
df = pd.read_csv("students.csv")

print("Preparing data...")
X = df[['maths', 'english', 'science', 'attendance']]
y = (df['maths'] + df['english'] + df['science']) / 3

print("Training model...")
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

print("Saving model...")
joblib.dump(model, "model.pkl")

print("✅ model.pkl created SUCCESSFULLY")
