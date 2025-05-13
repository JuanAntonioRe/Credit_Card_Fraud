import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# --- 5.1 Función extendida con validación cruzada ---
def evaluate_model(name, model, X, y, y_true, y_pred, y_proba):
    print(f"\n🔍 {name} Results:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    print("ROC-AUC Score:", roc_auc_score(y_true, y_proba))

    # --- Validación cruzada ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
    print(f"F1 CV Mean: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")

df = pd.read_csv('Dataset/intermediate/preprocess_data.csv')

df = df.dropna()

# --- 1. Selección de top 10 features ---
top_10_features = [
    'payments', 'purchases', 'credit_limit', 'minimum_payments', 'oneoff_purchases',
    'balance', 'cash_advance', 'cash_advance_trx', 'purchases_trx', 'installments_purchases'
]

# 1. Variables a usar (ejemplo, puedes ajustar)
X = df[top_10_features]
y = df['fraud']

# --- SPLIT: PRIMERO SEPARAR TRAIN/TEST ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Aumentamos la clase minoritaria (fraude = 1)
fraud_indices = y_train[y_train == 1].index
n_to_add = y_train.value_counts()[0] - y_train.value_counts()[1]
resampled_indices = np.random.choice(fraud_indices, size=n_to_add, replace=True)

X_resampled = pd.concat([X_train, X_train.loc[resampled_indices]])
y_resampled = pd.concat([y_train, y_train.loc[resampled_indices]])

# --- ESCALADO ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# --- 6. Random Forest ---
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, scoring='f1', cv=3, n_jobs=-1)
rf_grid.fit(X_train_scaled, y_resampled)
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test_scaled)
evaluate_model("Random Forest", rf_best, X_train_scaled, y_resampled, y_test, y_pred_rf, rf_best.predict_proba(X_test_scaled)[:, 1])

