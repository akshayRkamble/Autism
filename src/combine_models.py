import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

"""
Combine predictions from questionnaire and image models for autism screening.
 Loads train/test splits
 Trains best models
 Combines predictions and evaluates
"""

def load_questionnaire():
    npz_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Autism Screening", "questionnaire_train_test.npz"))
    data = np.load(npz_path, allow_pickle=True)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

    """Load train/test splits for questionnaire data."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        return data['X_train'], data['X_test'], data['y_train'], data['y_test']
    except Exception as e:
        print(f"Error loading questionnaire data: {e}")
        raise

def load_image_data():
    npz_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Image", "image_train_test.npz"))
    data = np.load(npz_path)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

    """Load train/test splits for image data."""
    try:
        data = np.load(npz_path)
        return data['X_train'], data['X_test'], data['y_train'], data['y_test']
    except Exception as e:
        print(f"Error loading image data: {e}")
        raise

def get_best_questionnaire_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_scaled, y_train)
    return model, scaler

    """Train and return best questionnaire model and scaler."""

def get_best_image_model(X_train, y_train, n_classes):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_scaled, y_train)
    return model, scaler

    """Train and return best image model and scaler."""

def combine_predictions(q_model, q_scaler, i_model, i_scaler, Xq, Xi):
    Xq_scaled = q_scaler.transform(Xq)
    Xi_scaled = i_scaler.transform(Xi)
    # Get class order for both models
    q_classes = q_model.classes_
    i_classes = i_model.classes_
    # Find common classes and their indices
    common_classes = np.intersect1d(q_classes, i_classes)
    q_indices = [np.where(q_classes == c)[0][0] for c in common_classes]
    i_indices = [np.where(i_classes == c)[0][0] for c in common_classes]
    # Get probabilities for common classes only
    q_pred = q_model.predict_proba(Xq_scaled)[:, q_indices]
    i_pred = i_model.predict_proba(Xi_scaled)[:, i_indices]
    # Average probabilities
    combined_prob = (q_pred + i_pred) / 2
    combined_pred = np.argmax(combined_prob, axis=1)
    # Map back to class labels
    combined_pred_labels = common_classes[combined_pred]
    return combined_pred_labels

    """Combine predictions from questionnaire and image models."""

def main():
    # Load data
    Xq_train, Xq_test, yq_train, yq_test = load_questionnaire()
    Xi_train, Xi_test, yi_train, yi_test = load_image_data()
    # Align test set sizes (use min length)
    min_len = min(len(Xq_test), len(Xi_test))
    Xq_test = Xq_test[:min_len]
    Xi_test = Xi_test[:min_len]
    yq_test = yq_test[:min_len]
    yi_test = yi_test[:min_len]
    # Align class counts (use intersection of unique classes)
    q_classes = np.unique(yq_train)
    i_classes = np.unique(yi_train)
    if not np.array_equal(q_classes, i_classes):
        print("Warning: Class labels differ between questionnaire and image data. Using intersection.")
        common_classes = np.intersect1d(q_classes, i_classes)
        # Filter test sets to only common classes
        mask = np.isin(yq_test, common_classes) & np.isin(yi_test, common_classes)
        Xq_test = Xq_test[mask]
        Xi_test = Xi_test[mask]
        yq_test = yq_test[mask]
        yi_test = yi_test[mask]
    n_classes = len(np.unique(yq_train))
    # Train best models (RandomForest for both)
    q_model, q_scaler = get_best_questionnaire_model(Xq_train, yq_train)
    i_model, i_scaler = get_best_image_model(Xi_train, yi_train, n_classes)
    # Combine predictions on test set
    combined_pred = combine_predictions(q_model, q_scaler, i_model, i_scaler, Xq_test, Xi_test)
    from sklearn.metrics import f1_score
    print("Combined Model Accuracy:", accuracy_score(yq_test, combined_pred))
    print("Combined Model F1 Score:", f1_score(yq_test, combined_pred, average='weighted'))
    print(classification_report(yq_test, combined_pred))

    """Main function to combine and evaluate predictions."""

if __name__ == "__main__":
    main()
