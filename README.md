# House Price Prediction

Live at: https://alfieeya.github.io/house-price-prediction/

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

## Possible next steps

- Try Gradient Boosting (XGBoost / LightGBM) for potentially better accuracy
- Add cross-validation curves to check for overfitting
- Build a simple Streamlit or Flask app around the saved model for a live demo
- Deploy it and add the link to your resume alongside the GitHub repo
