import pandas as pd
import numpy as np
import os

# Base directory for the project
BASE_DIR = "/home/jiwonjjeong/gk-lab/Automated-CV-Design"

# Path to the input file used in dimensionality reduction notebooks
MPSO_INPUT_FILE = os.path.join(BASE_DIR, "tests", "3_feature_selection", "mpso.csv")

def assign_classes(df, n_points_per_class=754, num_classes=3, start_label=0):
    """
    Assigns class labels to a DataFrame, matching the logic in dimensionality reduction notebooks.
    
    Args:
        df: The DataFrame to modify.
        n_points_per_class: Number of points in each class.
        num_classes: Total number of classes.
        start_label: The starting integer for class labels (e.g., 0 or 1).
        
    Returns:
        pd.DataFrame: DataFrame with the 'class' column assigned.
    """
    labels = []
    for i in range(num_classes):
        labels.extend([i + start_label] * n_points_per_class)
    
    # Ensure labels match the length of the DataFrame if it's different
    if len(labels) != len(df):
        print(f"Warning: Label count ({len(labels)}) does not match DataFrame row count ({len(df)}). Truncating or padding.")
        labels = (labels * (len(df) // len(labels) + 1))[:len(df)]
    
    df['class'] = labels
    return df

def get_mpso_data():
    """Returns the MPSO input data as a DataFrame."""
    return pd.read_csv(MPSO_INPUT_FILE)
