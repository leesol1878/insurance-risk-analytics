"""
EDA utilities for insurance risk analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_loss_ratio(df):
    """
    Calculate loss ratio = TotalClaims / TotalPremium
    """
    df = df.copy()
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    return df

def plot_numerical_distributions(df, numerical_cols, cols=3, figsize=(15, 12)):
    """
    Plot histograms for numerical columns
    """
    rows = (len(numerical_cols) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows > 1 else [axes] if cols > 1 else axes
    
    for i, col in enumerate(numerical_cols):
        if col in df.columns:
            axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_categorical_distributions(df, categorical_cols, cols=2, figsize=(12, 10)):
    """
    Plot bar charts for categorical columns
    """
    rows = (len(categorical_cols) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows > 1 else [axes] if cols > 1 else axes
    
    for i, col in enumerate(categorical_cols):
        if col in df.columns:
            value_counts = df[col].value_counts().head(10)  # Top 10 categories
            axes[i].bar(value_counts.index.astype(str), value_counts.values)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

def detect_outliers(df, column):
    """
    Detect outliers using IQR method
    
    Returns: outliers, lower_bound, upper_bound
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound