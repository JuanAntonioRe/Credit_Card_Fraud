import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib



# Evaluate model function with cross validation
def evaluate_model(name, model, X, y, y_true, y_pred, y_proba):
    print(f"\n🔍 {name} Results:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'modeling_output/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("ROC-AUC Score:", roc_auc_score(y_true, y_proba))

    # --- Validación cruzada ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
    print(f"F1 CV Mean: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")
    
# Features and target
X = pd.read_csv('Dataset/intermediate/X.csv')
y = pd.read_csv('Dataset/intermediate/y.csv')
y = y['fraud']

# Split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- Random Forest ----------
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_grid = GridSearchCV(rf, rf_params, scoring='f1', cv=3, n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test_scaled)
evaluate_model("Random Forest", rf_best, X_train_scaled, y_train, y_test, y_pred_rf, rf_best.predict_proba(X_test_scaled)[:, 1])


# Save model ---------------------------------------- 
joblib.dump(
        rf_best,
        f"modeling_output/b01_model_rf.joblib"
        )

print('\nModel saved')