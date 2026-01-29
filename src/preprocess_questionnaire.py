"""
Preprocess questionnaire data for autism screening.
 - Loads CSV/ARFF data
 - Handles missing values
 - Encodes categorical features
 - Saves processed data
"""

import pandas as pd
import os

def load_csv_data(filepath):
    """Load CSV or ARFF data from the given filepath."""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        raise
    df = pd.read_csv(filepath)
    return df

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_questionnaire(df):
    """Preprocess questionnaire DataFrame: handle missing values and encode categorical features."""
    # Drop rows with too many missing values
    df = df.dropna(thresh=int(0.7 * len(df.columns)))
    # Fill missing values for numerical columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(num_cols) > 0:
        imputer = SimpleImputer(strategy="mean")
        df[num_cols] = imputer.fit_transform(df[num_cols])
    # Fill missing values for categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    # Encode categorical columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df
    # Drop rows with too many missing values
    df = df.dropna(thresh=int(0.7*len(df.columns)))
    # Fill missing values for numerical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 0:
        imputer = SimpleImputer(strategy='mean')
        df[num_cols] = imputer.fit_transform(df[num_cols])
    # Fill missing values for categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    # Encode categorical columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

if __name__ == "__main__":
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Autism Screening", "Autism_Data.arff"))
    try:
        df = load_csv_data(csv_path)
        print("Original data shape:", df.shape)
        df_processed = preprocess_questionnaire(df)
        print("Processed data shape:", df_processed.shape)
        out_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Autism Screening", "Autism_Data_processed.csv"))
        df_processed.to_csv(out_csv, index=False)
        print(f"Saved processed data to {out_csv}")
    except Exception as e:
        print(f"Processing failed: {e}")
    df = load_csv_data(csv_path)
    print("Original data shape:", df.shape)
    df_processed = preprocess_questionnaire(df)
    print("Processed data shape:", df_processed.shape)
    out_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Autism Screening", "Autism_Data_processed.csv"))
    df_processed.to_csv(out_csv, index=False)
    print(f"Saved processed data to {out_csv}")
