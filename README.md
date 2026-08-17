# House Price Prediction

A machine learning model that predicts house prices based on input parameters
like median income, number of rooms, house age, and location.

> **Note on the dataset:** This project originally used the classic *Boston
> Housing* dataset, but it was removed from scikit-learn due to ethical
> concerns about some of its features. This version uses the **California
> Housing** dataset instead — it's the standard modern replacement and works
> the same way (regression on housing features → predict price).

## What it does

1. Loads the California Housing dataset (8 features, ~20,000 rows)
2. Explores the data (summary stats, correlation heatmap, price distribution)
3. Trains and compares two models:
   - Linear Regression
   - Random Forest Regressor (with a small hyperparameter grid search)
4. Evaluates both with RMSE, MAE, and R²
5. Saves the best model + scaler to disk (`.joblib` files)
6. Plots feature importance and actual-vs-predicted prices
7. Runs a demo prediction on a made-up sample house

## Features used

| Feature | Description |
|---|---|
| MedInc | Median income in the block group (in $10,000s) |
| HouseAge | Median house age |
| AveRooms | Average rooms per household |
| AveBedrms | Average bedrooms per household |
| Population | Block group population |
| AveOccup | Average household occupancy |
| Latitude / Longitude | Location |

Target: `PRICE` — median house value (in $100,000s)

## Setup

```bash
pip install -r requirements.txt
python house_price_prediction.py
```

This will print dataset stats and model metrics to the console, and save these
files in the project folder:

- `correlation_heatmap.png`
- `price_distribution.png`
- `feature_importance.png`
- `actual_vs_predicted.png`
- `house_price_model.joblib` (trained model)
- `scaler.joblib` (feature scaler, needed to preprocess new inputs the same way)

## A note on offline environments

`fetch_california_housing()` downloads data on first use, so it needs
internet access. If it can't reach the internet, the script automatically
falls back to a synthetic dataset with the same feature structure so the
pipeline still runs end-to-end. If you have internet access, it'll use the
real dataset automatically — no code changes needed.

## Using the saved model on new data

```python
import joblib
import pandas as pd

model = joblib.load("house_price_model.joblib")
scaler = joblib.load("scaler.joblib")

new_house = pd.DataFrame([{
    "MedInc": 5.0,
    "HouseAge": 20.0,
    "AveRooms": 6.0,
    "AveBedrms": 1.0,
    "Population": 1000.0,
    "AveOccup": 3.0,
    "Latitude": 34.05,
    "Longitude": -118.25,
}])

scaled = scaler.transform(new_house)
predicted_price = model.predict(scaled)[0] * 100000
print(f"Predicted price: ${predicted_price:,.2f}")
```

## Possible next steps

- Try Gradient Boosting (XGBoost / LightGBM) for potentially better accuracy
- Add cross-validation curves to check for overfitting
- Build a simple Streamlit or Flask app around the saved model for a live demo
- Deploy it and add the link to your resume alongside the GitHub repo
