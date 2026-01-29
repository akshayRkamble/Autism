import pickle
from train_image_models import load_image_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = load_image_data()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_scaled, y_train)
with open('../image_rf.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('../image_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print('Saved image RandomForest model and scaler.')
