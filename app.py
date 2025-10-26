from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/fake_profile_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [
            float(request.form["username_length"]),
            float(request.form["num_followers"]),
            float(request.form["num_following"]),
            float(request.form["num_posts"]),
            float(request.form["account_age_days"]),
            float(request.form["bio_length"]),
            float(request.form["profile_pic"]),
        ]
        scaled_data = scaler.transform([data])
        prediction = model.predict(scaled_data)[0]
        result = "Fake Profile ❌" if prediction == 1 else "Genuine Profile ✅"
        return render_template("result.html", result=result)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
