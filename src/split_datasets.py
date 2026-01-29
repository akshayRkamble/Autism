import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_questionnaire():
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Autism Screening", "Autism_Data_processed.csv"))
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[col for col in df.columns if 'Class' in col or 'ASD' in col])
    y = df[[col for col in df.columns if 'Class' in col or 'ASD' in col]].iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    np.savez(os.path.join(os.path.dirname(__file__), "..", "Autism Screening", "questionnaire_train_test.npz"), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print("Saved questionnaire train/test split.")

def split_images():
    npz_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Image", "train_image_features.npz"))
    data = np.load(npz_path)
    X, y = data['X'], data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    np.savez(os.path.join(os.path.dirname(__file__), "..", "Image", "image_train_test.npz"), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print("Saved image train/test split.")

if __name__ == "__main__":
    split_questionnaire()
    split_images()
