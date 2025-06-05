# Library Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import tensorflow as tf
from tensorflow.keras import layers

#------------------------------------------------------------------------------------------------

# Data Preperation
## 1.
def load_and_prepare_data(file_path, test_size=0.20):
    """
    Load the dataset, clean it by removing irrelevant columns, and split it into
    training+validation and testing sets.
    """
    # Load data
    df = pd.read_csv(file_path)

    # Drop non-feature columns
    X_full = df.drop(columns=['state', 'net in', 'net out', 'net total'])
    y_full = df['net total']

    # Train-test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(X_full, y_full, test_size=test_size)

    return X_trainval, X_test, y_trainval, y_test

## 2.
def standardization(ds, means=None, stds=None):
    """
    Perform Z‐score standardization on the df
    """
    if means is None or stds is None:
        means = ds.mean()
        stds  = ds.std(ddof=0)    # population standard deviation
    ds_scaled = (ds - means) / stds
    return ds_scaled, means, stds

#------------------------------------------------------------------------------------------------

# Feature Selection (Running Lasso) (Target: Net Migration Rate)
def lasso_feature_selection(X, y, cv=5, n_alphas=150):
    """
    Perform Lasso regression with cross-validation to select optimal alpha
    and identify important features with non-zero coefficients.
    Params:
        cv : Number of cross-validation folds.
        n_alphas : Number of alphas along the regularization path.
    Returns:
        lasso_cv: The fitted LassoCV model.
        selected_features: Series of selected features and their coefficients (non-zero only).
    """
    # Fit LassoCV
    lasso_cv = LassoCV(cv=cv, max_iter=10000, n_alphas=n_alphas)
    lasso_cv.fit(X, y)

    # Extract non-zero coefficients
    coef_series = pd.Series(lasso_cv.coef_, index=X.columns)
    selected_features = coef_series[coef_series != 0]

    # Print results
    print(f"Optimal alpha is {lasso_cv.alpha_:.5f}")
    print("Selected features and coefficients:")
    print(selected_features)

    return lasso_cv, selected_features

#------------------------------------------------------------------------------------------------

# Model Construction (Target: Net Migration Rate)

## 1. Random Forest
def grid_search_random_forest(X_train, y_train, X_val, y_val, 
                               n_estimators_list, 
                               max_depth_list):
    """
    Perform grid search over n_estimators and max_depth for Random Forest.
    Returns:
        rf_df: grid search results, sorted by RMSE.
    """
    results = []
    for n_est in n_estimators_list:
        for depth in max_depth_list:
            rf = RandomForestRegressor(n_estimators=n_est, max_depth=depth)
            rf.fit(X_train, y_train)
            y_val_pred = rf.predict(X_val)
            rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
            results.append({
                'n_estimators': n_est,
                'max_depth': depth,
                'rmse_val': rmse_val
            })
    rf_df = pd.DataFrame(results).sort_values('rmse_val').reset_index(drop=True)
    print("Random Forest grid-search results:")
    print(rf_df)
    return rf_df

def fit_best_random_forest(X_trainval, y_trainval, X_test, y_test, param):
    """
    Fit Random Forest using best parameters and evaluate on test set.
    Params:
        best_params : Dictionary with 'n_estimators' and 'max_depth' as keys.
    Returns:
        final_model: Trained model.
        rmse_test: RMSE on test set.
    """
    best_rf_params = param.iloc[0]
    best_n = int(best_rf_params['n_estimators'])
    best_depth = int(best_rf_params['max_depth'])

    final_model = RandomForestRegressor(n_estimators=best_n, max_depth=best_depth)
    final_model.fit(X_trainval, y_trainval)

    y_test_pred = final_model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(f"Random Forest Test RMSE: {rmse_test:.5f}")

    return final_model, rmse_test


## 2. Gradient Boosting
def grid_search_gradient_boosting(X_train, y_train, X_val, y_val, learning_rate_list, max_depth_list,n_estimators_list):
    """
    Perform grid search over learning_rate, max_depth, and n_estimators.
    Returns:
        gbm_df: Grid search results sorted by RMSE on validation set.
    """
    results = []
    for lr in learning_rate_list:
        for depth in max_depth_list:
            for n_est in n_estimators_list:
                gbm = GradientBoostingRegressor(learning_rate=lr,
                                                max_depth=depth,
                                                n_estimators=n_est)
                gbm.fit(X_train, y_train)
                y_val_pred = gbm.predict(X_val)
                rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

                results.append({'learning_rate': lr,
                                'max_depth': depth,
                                'n_estimators': n_est,
                                'rmse_val': rmse_val})
    gbm_df = pd.DataFrame(results).sort_values('rmse_val').reset_index(drop=True)
    print("Gradient Boosting grid search results:")
    print(gbm_df)
    return gbm_df

def fit_best_gradient_boosting(X_trainval, y_trainval, X_test, y_test, param_df):
    """
    Fit Gradient Boosting using best parameters from grid search and evaluate on test set.
    Params:
        param_df: Output from grid_search_gradient_boosting (gbm_df)
    Returns:
        final_model: trained GradientBoostingRegressor
        rmse_test: The RMSE value
    """
    best_params = param_df.iloc[0]
    best_lr = float(best_params['learning_rate'])
    best_depth = int(best_params['max_depth'])
    best_n = int(best_params['n_estimators'])

    final_model = GradientBoostingRegressor(learning_rate=best_lr,
                                            max_depth=best_depth,
                                            n_estimators=best_n)
    final_model.fit(X_trainval, y_trainval)

    y_test_pred = final_model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(f"Gradient Boosting Test RMSE: {rmse_test:.5f}")
    return final_model, rmse_test


# 3. MLP model
def train_evaluate_mlp(X_trainval_sel_scaled, y_trainval_sel, X_test_scaled, y_test, input_dim=8, epochs=100, batch_size=64):
    """
    Build, train, and evaluate an MLP model.
    Params:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
    Returns:
    mlp_model: Trained Keras model
    test_rmse: RMSE on test set
    """
    # Build the MLP model:
    mlp_model = tf.keras.models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2), # add non-linear layer
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(8, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='linear')
        ])

    # Compile the MLP model
    mlp_model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])

    # Train on dataset
    history = mlp_model.fit(X_trainval_sel_scaled, y_trainval_sel, batch_size=batch_size, epochs=epochs)

    # Evaluate on test set
    rmse_test_mlp = mlp_model.evaluate(X_test_scaled, y_test)
    return mlp_model, rmse_test_mlp[1], history
