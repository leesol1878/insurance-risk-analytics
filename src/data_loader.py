"""
Data loading utilities for insurance risk analytics
"""

import pandas as pd
from pathlib import Path
import os

def load_insurance_data(file_path=None):
    """
    Load the insurance dataset from the data folder
    
    Parameters:
    file_path: str, optional - path to the CSV file
    
    Returns:
    pandas DataFrame with the insurance data
    """
    if file_path is None:
        # Go up one level from src to root, then into data
        root_dir = Path(__file__).parent.parent
        data_dir = root_dir / "data"
        
        # Look for any CSV file in the data folder
        csv_files = list(data_dir.glob("*.csv"))
        
        if len(csv_files) == 0:
            # If no CSV, look for txt files that might be CSV
            csv_files = list(data_dir.glob("*.txt"))
        
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV or TXT files found in {data_dir}")
        
        file_path = csv_files[0]
        print(f"Loading file: {file_path.name}")
    
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully! Shape: {df.shape}")
    return df

def get_basic_info(df):
    """
    Get basic information about the dataset
    
    Parameters:
    df: pandas DataFrame
    
    Returns:
    Dictionary with basic info
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    return info