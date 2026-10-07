import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Suppress some verbose warnings for clean CLI output
warnings.filterwarnings('ignore')

from features import TitleFeatureExtractor

def run_milestone3():
    print("Loading dataset and preprocessing pipeline...")
    try:
        df = pd.read_csv('data/dataset.csv')
        preprocessor = joblib.load('models/preprocessor.pkl')
    except Exception as e:
        print(f"Error loading files: {e}. Please ensure Milestones 1 and 2 were run.")
        return

    # Using true_fair_value as our ground-truth target
    X = df.drop(columns=['true_fair_value'])
    y = df['true_fair_value']
    
    # We still perform a train_test_split so we can save a pure holdout set for Milestone 4 (Residual Analysis)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Applying preprocessing to training data...")
    X_train = preprocessor.transform(X_train_raw)
    
    # Define models to benchmark
    models = {
        'Ridge Regression': Ridge(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1)
    }
    
    scoring = {
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2'
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n--- Benchmarking Models (5-Fold CV) ---")
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring, n_jobs=-1)
        
        # Extract mean absolute error, RMSE and R2 scores
        mae = -cv_results['test_mae'].mean()
        rmse = -cv_results['test_rmse'].mean()
        r2 = cv_results['test_r2'].mean()
        
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"  > MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R2: {r2:.4f}")

    # Determine best model based on MAE (Mean Absolute Error)
    best_model_name = min(results, key=lambda k: results[k]['MAE'])
    print(f"\nBest model selected: {best_model_name} (Lowest MAE)")
    
    print(f"\n--- Hyperparameter Tuning {best_model_name} ---")
    
    # Assign specific tuning grids based on the winning architecture
    if best_model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        }
        base_model = XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1)
    elif best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    else:
        param_grid = {'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]}
        base_model = Ridge(random_state=42)
        
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    print("Running Grid Search over hyperparameters...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best Parameters Found: {grid_search.best_params_}")
    
    print("\n--- Final Model Training & Saving ---")
    
    # Save the finalized, tuned model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    
    # We also save the test splits for Milestone 4's evaluation plot
    X_test_raw.to_pickle('data/X_test_raw.pkl')
    y_test.to_pickle('data/y_test.pkl')
    
    print("Final tuned model successfully saved to models/best_model.pkl")
    print("Test splits saved to data/ for final evaluation.")

if __name__ == '__main__':
    run_milestone3()
