import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
import joblib
import os

from features import TitleFeatureExtractor

def get_preprocessing_pipeline():
    # 1. Text/Title Feature Extraction
    title_pipeline = Pipeline([
        ('extractor', TitleFeatureExtractor())
    ])
    
    # 2. Categorical Encoding (OneHot)
    categorical_features = ['platform', 'condition']
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 3. Numerical Scaling and Transformation
    # We apply a log1p transform to seller_ratings_count to handle right-skewness
    log_transformer = FunctionTransformer(np.log1p, feature_names_out='one-to-one')
    
    numerical_features_log = ['seller_ratings_count']
    numerical_features_standard = ['seller_feedback_pct']
    
    numerical_pipeline = ColumnTransformer([
        ('log_scale', Pipeline([
            ('log', log_transformer),
            ('scaler', StandardScaler())
        ]), numerical_features_log),
        ('standard_scale', StandardScaler(), numerical_features_standard)
    ])
    
    # Assemble the final ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('title', title_pipeline, ['title']),
            ('cat', categorical_pipeline, categorical_features),
            ('num', numerical_pipeline, numerical_features_log + numerical_features_standard)
        ],
        # Drop columns not explicitly handled (like listed_price and true_fair_value)
        # to prevent data leakage during model training!
        remainder='drop' 
    )
    
    return preprocessor

def run_milestone2():
    print("Loading data from data/dataset.csv...")
    try:
        df = pd.read_csv('data/dataset.csv')
    except FileNotFoundError:
        print("Error: data/dataset.csv not found. Did you run Milestone 1?")
        return
        
    print(f"Data loaded. Original shape: {df.shape}")
    
    # We separate features from target. 
    # Notice we keep listed_price in the DataFrame for now, but our ColumnTransformer
    # will automatically drop it because of remainder='drop', preventing leakage.
    X = df.drop(columns=['true_fair_value']) 
    
    preprocessor = get_preprocessing_pipeline()
    
    print("Fitting and transforming data through the pipeline...")
    X_processed = preprocessor.fit_transform(X)
    
    # Generate mock column names for visualization
    cat_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['platform', 'condition'])
    title_extractor = preprocessor.named_transformers_['title'].named_steps['extractor']
    title_names = [f'has_{kw}' for kw in title_extractor.keywords]
    all_feature_names = title_names + list(cat_names) + ['seller_ratings_count'] + ['seller_feedback_pct']
    
    print(f"Transformation complete. Processed feature matrix shape: {X_processed.shape}")
    print(f"Extracted Features: {all_feature_names[:5]} ... {all_feature_names[-5:]}")
    
    # Save the pipeline artifact for Milestone 3 & 4
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("Anti-leakage pipeline successfully saved to models/preprocessor.pkl")

if __name__ == '__main__':
    run_milestone2()
