import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from features import TitleFeatureExtractor

def run_evaluation():
    print("Loading artifacts and test splits...")
    try:
        preprocessor = joblib.load('models/preprocessor.pkl')
        model = joblib.load('models/best_model.pkl')
        X_test_raw = pd.read_pickle('data/X_test_raw.pkl')
        y_test = pd.read_pickle('data/y_test.pkl')
    except Exception as e:
        print(f"Error loading files: {e}. Please ensure Milestone 3 completed successfully.")
        return
        
    print("Generating predictions...")
    X_test_processed = preprocessor.transform(X_test_raw)
    y_pred = model.predict(X_test_processed)
    
    # Create evaluation directory
    os.makedirs('plots', exist_ok=True)
    
    # 1. Actual vs Predicted Fair Value Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Fair Value ($)')
    plt.ylabel('Predicted Fair Value ($)')
    plt.title('True Fair Value vs. Predicted Fair Value')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/actual_vs_predicted.png')
    print("Saved 'plots/actual_vs_predicted.png'")
    
    # 2. Residual Distribution Plot (Error in Fair Value Prediction)
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='purple', bins=40)
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel('Prediction Error ($)')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution (Actual - Predicted Fair Value)')
    plt.savefig('plots/residual_distribution.png')
    print("Saved 'plots/residual_distribution.png'")
    
    # 3. Feature Importances (if applicable)
    if hasattr(model, 'feature_importances_'):
        # Reconstruct feature names manually since Pipeline can obfuscate them
        cat_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['platform', 'condition'])
        title_extractor = preprocessor.named_transformers_['title'].named_steps['extractor']
        title_names = [f'has_{kw}' for kw in title_extractor.keywords]
        feature_names = title_names + list(cat_names) + ['seller_ratings_count', 'seller_feedback_pct']
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:] # Top 15
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], color='green', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig('plots/feature_importances.png')
        print("Saved 'plots/feature_importances.png'")

    print("\nEvaluation successfully completed! Check the 'plots' folder for your graphs.")

if __name__ == '__main__':
    run_evaluation()
