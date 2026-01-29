import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

"""
Train and evaluate models for questionnaire-based autism screening.
 Loads train/test splits
 Trains multiple classifiers and ensemble
 Saves best model and scaler
"""

# Load train/test splits for questionnaire data
def load_questionnaire():
    npz_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Autism Screening", "questionnaire_train_test.npz"))
    data = np.load(npz_path, allow_pickle=True)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

    """Load train/test splits for questionnaire data."""
    npz_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Autism Screening", "questionnaire_train_test.npz"))
    try:
        data = np.load(npz_path, allow_pickle=True)
        return data['X_train'], data['X_test'], data['y_train'], data['y_test']
    except Exception as e:
        print(f"Error loading questionnaire data: {e}")
        raise

# Train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'GradientBoosting': GradientBoostingClassifier(),
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
    # Ensemble
    ensemble = VotingClassifier(estimators=[(n, m) for n, m in models.items()], voting='soft')
    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict(X_test)
    acc_ens = accuracy_score(y_test, y_pred_ens)
    results['Ensemble'] = acc_ens
    print(f"Ensemble Accuracy: {acc_ens:.4f}")
    print(classification_report(y_test, y_pred_ens))
    return results

    """Train and evaluate multiple classifiers and an ensemble."""

if __name__ == "__main__":
    import pickle
    X_train, X_test, y_train, y_test = load_questionnaire()
    print("Training on questionnaire data...")
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    print("All results:", results)
    # Save best model (RandomForest) and scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train_scaled, y_train)
    with open('questionnaire_rf.pkl', 'wb') as f:
        pickle.dump(rf, f)
    with open('questionnaire_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print('Saved questionnaire RandomForest model and scaler.')

    try:
        X_train, X_test, y_train, y_test = load_questionnaire()
        print("Training on questionnaire data...")
        results = train_and_evaluate(X_train, X_test, y_train, y_test)
        print("All results:", results)
        # Save best model (RandomForest) and scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train_scaled, y_train)
        with open('questionnaire_rf.pkl', 'wb') as f:
            pickle.dump(rf, f)
        with open('questionnaire_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print('Saved questionnaire RandomForest model and scaler.')
    except Exception as e:
        print(f"Training or saving failed: {e}")
