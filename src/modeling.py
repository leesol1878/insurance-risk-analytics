"""
Statistical Modeling Module for Insurance Risk Analytics
Contains reusable functions for model training, evaluation, and interpretation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Note: XGBoost requires separate installation: pip install xgboost
# Uncomment below if xgboost is installed
# from xgboost import XGBRegressor, XGBClassifier

def prepare_features(df, target_col, exclude_cols=None):
    """
    Prepare features for modeling
    
    Parameters:
    df: DataFrame
    target_col: name of target column
    exclude_cols: list of columns to exclude (e.g., IDs)
    
    Returns:
    X: features DataFrame
    y: target Series
    """
    if exclude_cols is None:
        exclude_cols = []
    
    exclude_cols = exclude_cols + [target_col]
    X = df.drop(columns=exclude_cols, errors='ignore')
    y = df[target_col]
    
    return X, y

def identify_column_types(df):
    """
    Identify numerical and categorical columns
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return numerical_cols, categorical_cols

def create_preprocessor(numerical_cols, categorical_cols):
    """
    Create preprocessing pipeline for numerical and categorical columns
    """
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor

def train_regression_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate regression models for claim severity
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    # Add XGBoost only if available
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    except ImportError:
        print("XGBoost not available. Skipping...")
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        results.append({
            'Model': name,
            'Train RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'Train R²': r2_score(y_train, y_pred_train),
            'Test R²': r2_score(y_test, y_pred_test)
        })
    
    return results, trained_models

def train_classification_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate classification models for claim probability
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    # Add XGBoost only if available
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
    except ImportError:
        print("XGBoost not available. Skipping...")
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        results.append({
            'Model': name,
            'Train Accuracy': accuracy_score(y_train, y_pred_train),
            'Test Accuracy': accuracy_score(y_test, y_pred_test),
            'Precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        })
    
    return results, trained_models

def calculate_risk_premium(claim_probability, expected_severity, expense_loading=0.15, profit_margin=0.10):
    """
    Calculate risk-based premium using the formula:
    Premium = P(Claim) × Expected Severity × (1 + Expense Loading + Profit Margin)
    """
    base_premium = claim_probability * expected_severity
    total_premium = base_premium * (1 + expense_loading + profit_margin)
    return total_premium