import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib, os

os.makedirs("model", exist_ok=True)
df = pd.read_csv("dataset/social_profiles.csv")

X = df.drop("fake", axis=1)
y = df["fake"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(rf, "model/fake_profile_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("✅ Model and scaler saved in 'model/' folder.")
