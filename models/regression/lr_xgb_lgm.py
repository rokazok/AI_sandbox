# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import statsmodels.api as sm


def main():
    """
    This script trains and compares XGBoost, LightGBM, and Linear Regression models on the California Housing dataset.
    Explicit function arguments are used in all model calls.
    Detailed comments are provided throughout.

    Data: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
    """

    # Load the California Housing dataset
    california = fetch_california_housing(as_frame=True)
    X = california.data
    y = california.target

    # Split the dataset into training and testing sets
    # test_size=0.2 means 20% test, random_state=42 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Linear Regression (with feature scaling)
    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and fit the linear regression model
    lr_model = LinearRegression()
    lr_model.fit(X=X_train_scaled, y=y_train)
    # Predict on test set
    lr_preds = lr_model.predict(X_test_scaled)
    # Evaluate performance
    lr_mse = mean_squared_error(y_true=y_test, y_pred=lr_preds)
    lr_r2 = r2_score(y_true=y_test, y_pred=lr_preds)

    # Comments:
    # - StandardScaler is used to scale features for the linear regression model.
    # - Scaling is fit on training data and applied to both train and test sets.
    # - Tree-based models (XGBoost, LightGBM) do not require feature scaling.

    # 1b. Statsmodels Linear Regression (with feature scaling)

    # Add intercept manually for statsmodels
    X_train_sm = sm.add_constant(X_train_scaled)
    X_test_sm = sm.add_constant(X_test_scaled)

    # Fit the OLS model
    sm_lr_model = sm.OLS(endog=y_train, exog=X_train_sm)
    sm_lr_results = sm_lr_model.fit()
    # Predict on test set
    sm_lr_preds = sm_lr_results.predict(X_test_sm)
    # Evaluate performance
    sm_lr_mse = mean_squared_error(y_true=y_test, y_pred=sm_lr_preds)
    sm_lr_r2 = r2_score(y_true=y_test, y_pred=sm_lr_preds)

    # 2. XGBoost Regression
    # Create DMatrix objects for XGBoost
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_test = xgb.DMatrix(X_test, label=y_test)
    # Set parameters explicitly
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'eta': 0.1,
        'seed': 42
    }
    # Train the model
    xgb_model = xgb.train(
        params=xgb_params,
        dtrain=xgb_train,
        num_boost_round=100
    )
    # Predict on test set
    xgb_preds = xgb_model.predict(xgb_test)
    # Evaluate performance
    xgb_mse = mean_squared_error(y_true=y_test, y_pred=xgb_preds)
    xgb_r2 = r2_score(y_true=y_test, y_pred=xgb_preds)

    # 3. LightGBM Regression
    # Prepare datasets for LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
    # Set parameters explicitly
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'seed': 42
    }
    # Train the model
    lgb_model = lgb.train(
        params=lgb_params,
        train_set=lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_test],
        callbacks=[lgb.log_evaluation(period=10)]
    )
    # Predict on test set
    lgb_preds = lgb_model.predict(X_test)
    # Evaluate performance
    lgb_mse = mean_squared_error(y_true=y_test, y_pred=lgb_preds)
    lgb_r2 = r2_score(y_true=y_test, y_pred=lgb_preds)

    # Print comparison of results
    print("Model Performance Comparison:")
    print(f"Linear Regression (sklearn):     MSE = {lr_mse:.4f}, R2 = {lr_r2:.4f}")
    print(f"Linear Regression (statsmodels): MSE = {sm_lr_mse:.4f}, R2 = {sm_lr_r2:.4f}")
    print(f"XGBoost Regression:              MSE = {xgb_mse:.4f}, R2 = {xgb_r2:.4f}")
    print(f"LightGBM Regression:             MSE = {lgb_mse:.4f}, R2 = {lgb_r2:.4f}")

    # Create a table comparing true prices and predictions from all models
    results_df = pd.DataFrame({
        'True Price': y_test.values if hasattr(y_test, 'values') else y_test,
        'Linear Regression (sklearn)': lr_preds,
        'Linear Regression (statsmodels)': sm_lr_preds,
        'XGBoost': xgb_preds,
        'LightGBM': lgb_preds
    })

    # Show a sample of the comparison table
    print("\nSample predictions vs true prices:")
    print(results_df.head(10))

    # Comments:
    # - All models are trained and evaluated on the same California Housing dataset.
    # - Statsmodels OLS requires manual addition of the intercept (constant column).
    # - Feature scaling is applied before fitting both sklearn and statsmodels linear models.
    # - Function arguments are explicitly specified for model constructors and train_test_split.
    # - Performance is compared using Mean Squared Error (MSE) and R2 score.


if __name__ == "__main__":
    main()
