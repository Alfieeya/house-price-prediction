"""
House Price Prediction - Machine Learning Project
----------------------------------------------------
Predicts median house prices based on input parameters such as
median income, average rooms, house age, population, and location.

Dataset: California Housing (scikit-learn built-in)
Note: The original "Boston Housing" dataset was removed from scikit-learn
due to ethical concerns with some of its features. California Housing is
the standard modern replacement and works the same way for this project.

Models compared: Linear Regression, Random Forest Regressor
Metrics: RMSE, MAE, R^2 Score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42


def generate_synthetic_housing_data(n_samples=2000, random_state=RANDOM_STATE):
    """
    Generate a synthetic housing dataset with the same feature structure as
    the California Housing dataset. Used as an offline fallback when
    fetch_california_housing() can't reach the internet to download data.

    If you have internet access, just call load_data() normally and it will
    use the real dataset instead of this synthetic one.
    """
    rng = np.random.default_rng(random_state)

    med_inc = rng.gamma(shape=5, scale=1.0, size=n_samples)          # median income ($10,000s)
    house_age = rng.uniform(1, 52, n_samples)                        # median house age
    ave_rooms = rng.normal(6, 1.2, n_samples).clip(2, 12)             # avg rooms/household
    ave_bedrms = (ave_rooms * rng.uniform(0.15, 0.25, n_samples))     # avg bedrooms/household
    population = rng.uniform(200, 5000, n_samples)                   # block population
    ave_occup = rng.normal(3, 0.7, n_samples).clip(1, 8)              # avg household occupancy
    latitude = rng.uniform(32.5, 42.0, n_samples)                    # CA-like latitude range
    longitude = rng.uniform(-124.3, -114.3, n_samples)               # CA-like longitude range

    # Simulate a realistic price relationship + noise (target in $100,000s)
    price = (
        0.45 * med_inc
        + 0.02 * ave_rooms
        - 0.01 * house_age
        - 0.03 * ave_occup
        + 0.5
        + rng.normal(0, 0.4, n_samples)
    ).clip(0.15, 5.0)

    df = pd.DataFrame({
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
        "PRICE": price,
    })
    return df


def load_data():
    """
    Load the California Housing dataset into a pandas DataFrame.
    Falls back to a synthetic dataset with the same feature structure
    if there's no internet access to download the real data.
    """
    try:
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        df.rename(columns={"MedHouseVal": "PRICE"}, inplace=True)
        print("Loaded real California Housing dataset.")
    except Exception as e:
        print(f"Could not download California Housing dataset ({e}).")
        print("Falling back to a synthetic dataset with the same feature structure.")
        df = generate_synthetic_housing_data()
    return df


def explore_data(df):
    """Print basic info and save a correlation heatmap + price distribution plot."""
    print("\n--- Dataset Shape ---")
    print(df.shape)

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Summary Statistics ---")
    print(df.describe())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.close()

    # Price distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["PRICE"], kde=True, bins=40, color="steelblue")
    plt.title("Distribution of House Prices (in $100,000s)")
    plt.xlabel("Price")
    plt.tight_layout()
    plt.savefig("price_distribution.png", dpi=150)
    plt.close()

    print("\nSaved plots: correlation_heatmap.png, price_distribution.png")


def preprocess(df):
    """Split into features/target, then train/test split and scale features."""
    X = df.drop("PRICE", axis=1)
    y = df["PRICE"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


def evaluate_model(name, model, X_test, y_test):
    """Compute and print evaluation metrics for a trained model."""
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n--- {name} ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R^2:  {r2:.4f}")

    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train a Random Forest with a small grid search for key hyperparameters."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"\nBest Random Forest params: {grid.best_params_}")
    return grid.best_estimator_


def plot_feature_importance(model, feature_names):
    """Save a feature importance bar chart (Random Forest only)."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], color="seagreen")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("Saved plot: feature_importance.png")


def plot_predictions(model, X_test, y_test, name):
    """Save a scatter plot of actual vs predicted prices."""
    preds = model.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.3, color="darkorange")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted Price ({name})")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    plt.close()
    print("Saved plot: actual_vs_predicted.png")


def predict_new_sample(model, scaler, feature_names):
    """Demo: predict price for one made-up house."""
    sample = pd.DataFrame([{
        "MedInc": 5.0,       # median income (in $10,000s)
        "HouseAge": 20.0,    # median house age
        "AveRooms": 6.0,     # average rooms per household
        "AveBedrms": 1.0,    # average bedrooms per household
        "Population": 1000.0,
        "AveOccup": 3.0,     # average household occupancy
        "Latitude": 34.05,
        "Longitude": -118.25,
    }])[feature_names]

    sample_scaled = scaler.transform(sample)
    predicted_price = model.predict(sample_scaled)[0]
    print(f"\n--- Sample Prediction ---")
    print(f"Input: {sample.to_dict(orient='records')[0]}")
    print(f"Predicted Price: ${predicted_price * 100000:,.2f}")


def main():
    df = load_data()
    explore_data(df)

    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    lr_model = train_linear_regression(X_train, y_train)
    lr_results = evaluate_model("Linear Regression", lr_model, X_test, y_test)

    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model("Random Forest", rf_model, X_test, y_test)

    # Pick the better model based on R^2
    best_model, best_name = (rf_model, "Random Forest") if rf_results["r2"] > lr_results["r2"] else (lr_model, "Linear Regression")
    print(f"\nBest model: {best_name}")

    plot_feature_importance(rf_model, feature_names)
    plot_predictions(best_model, X_test, y_test, best_name)

    # Save model + scaler for reuse
    joblib.dump(best_model, "house_price_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    print("\nSaved model: house_price_model.joblib")
    print("Saved scaler: scaler.joblib")

    predict_new_sample(best_model, scaler, feature_names)


if __name__ == "__main__":
    main()
