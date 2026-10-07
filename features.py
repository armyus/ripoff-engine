import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TitleFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer to extract binary keyword flags from raw listing titles."""
    def __init__(self):
        self.keywords = ['cib', 'box', 'tested', 'scratch', 'flaw', 'working', 'loose', 'rare']
        
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        # Ensure we are working with a pandas Series for string operations
        if isinstance(X, pd.DataFrame):
            titles = X.iloc[:, 0]
        else:
            titles = pd.Series(X)
            
        features = pd.DataFrame(index=titles.index)
        titles_lower = titles.astype(str).str.lower()
        
        for kw in self.keywords:
            features[f'has_{kw}'] = titles_lower.str.contains(kw, na=False).astype(float)
            
        return features
