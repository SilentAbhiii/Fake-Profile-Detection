# Fake Profile Detection in Social Networks

## Overview
This project detects fake social media profiles using a Random Forest Classifier trained on synthetic data.

## Folder Structure
```
fake-profile-detection/
│
├── dataset/
│   └── social_profiles.csv
├── model/
│   └── fake_profile_model.pkl
├── static/
│   └── style.css
├── templates/
│   ├── index.html
│   └── result.html
├── app.py
└── train_model.py
```

## Instructions
1. Install dependencies:
   ```bash
   pip install flask scikit-learn pandas joblib
   ```
2. Train the model:
   ```bash
   python train_model.py
   ```
3. Run Flask app:
   ```bash
   python app.py
   ```
4. Visit: http://127.0.0.1:5000/

Dataset: synthetic data of 5000 profiles.
