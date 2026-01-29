import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# Load train/test splits for image data
def load_image_data():
    npz_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Image", "image_train_test.npz"))
    data = np.load(npz_path)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

# Train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Scale features for better convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=3000, solver='lbfgs'),
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'GradientBoosting': GradientBoostingClassifier(),
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier()
    }
    results = {}
    for name, model in models.items():
        if name in ['LogisticRegression', 'SVM', 'KNN']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.preprocessing import StandardScaler
        import os

        # CNN imports
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        from tensorflow.keras.utils import to_categorical

        def load_image_data():
            npz_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Image", "image_train_test.npz"))
            data = np.load(npz_path)
            return data['X_train'], data['X_test'], data['y_train'], data['y_test']

        def train_and_evaluate(X_train, X_test, y_train, y_test, n_classes):
            # Scale data for SVM, KNN, Logistic Regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {
                'LogisticRegression': LogisticRegression(max_iter=2000),
                'RandomForest': RandomForestClassifier(n_estimators=100),
                'GradientBoosting': GradientBoostingClassifier(),
                'SVM': SVC(probability=True),
                'KNN': KNeighborsClassifier()
            }
            results = {}
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)
                results[name] = acc
                print(f"{name} Accuracy: {acc:.4f}")
                print(classification_report(y_test, y_pred))
            # Ensemble
            ensemble = VotingClassifier(estimators=[(n, m) for n, m in models.items()], voting='soft')
            ensemble.fit(X_train_scaled, y_train)
            y_pred_ens = ensemble.predict(X_test_scaled)
            acc_ens = accuracy_score(y_test, y_pred_ens)
            results['Ensemble'] = acc_ens
            print(f"Ensemble Accuracy: {acc_ens:.4f}")
            print(classification_report(y_test, y_pred_ens))

            # CNN
            print("\nTraining simple CNN...")
            img_size = int(np.sqrt(X_train.shape[1]))
            X_train_cnn = X_train.reshape(-1, img_size, img_size, 1) / 255.0
            X_test_cnn = X_test.reshape(-1, img_size, img_size, 1) / 255.0
            y_train_cat = to_categorical(y_train, num_classes=n_classes)
            y_test_cat = to_categorical(y_test, num_classes=n_classes)
            cnn = Sequential([
                Conv2D(16, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
                MaxPooling2D(2,2),
                Conv2D(32, (3,3), activation='relu'),
                MaxPooling2D(2,2),
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(n_classes, activation='softmax')
            ])
            cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            cnn.fit(X_train_cnn, y_train_cat, epochs=10, batch_size=32, validation_split=0.1, verbose=2)
            cnn_eval = cnn.evaluate(X_test_cnn, y_test_cat, verbose=0)
            print(f"CNN Accuracy: {cnn_eval[1]:.4f}")
            results['CNN'] = cnn_eval[1]
            return results

        if __name__ == "__main__":
            import pickle
            X_train, X_test, y_train, y_test = load_image_data()
            n_classes = len(np.unique(y_train))
            print("Training on image data...")
            results = train_and_evaluate(X_train, X_test, y_train, y_test, n_classes)
            print("All results:", results)
            # Save best model (RandomForest) and scaler
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(X_train_scaled, y_train)
            with open('image_rf.pkl', 'wb') as f:
                pickle.dump(rf, f)
            with open('image_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            print('Saved image RandomForest model and scaler.')
