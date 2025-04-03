import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from credit_risk.config import PROCESSED_DATA_DIR, REPORTS_DIR
from pathlib import Path
from loguru import logger

# Load the processed dataset
input_path = PROCESSED_DATA_DIR / "normalized_data.csv"
df = pd.read_csv(input_path)
X = df.drop(columns=["default"])
y = df["default"]

logger.info("===== COMPREHENSIVE MODEL EVALUATION =====")
logger.info("Scaling dataset...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

logger.info("Training models on full dataset...")
log_reg = LogisticRegression()
rf = RandomForestClassifier(random_state=42)
iso_forest = IsolationForest(random_state=42)

log_reg.fit(X_scaled, y)
rf.fit(X_scaled, y)
iso_forest.fit(X_scaled[y == 0])

# Predict
lr_probs = log_reg.predict_proba(X_scaled)[:, 1]
rf_probs = rf.predict_proba(X_scaled)[:, 1]
if_scores = -iso_forest.decision_function(X_scaled)
if_probs = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())

results_df = pd.DataFrame({
    "actual": y,
    "lr_prob": lr_probs,
    "rf_prob": rf_probs,
    "if_prob": if_probs
})

model_outputs_path = REPORTS_DIR / "model_outputs.csv"
results_df.to_csv(model_outputs_path, index=False)
logger.success(f"Model probabilities saved to {model_outputs_path}")

# AUPRC calculation
def calculate_auprc(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

logger.info("AUPRC Scores")
logger.info(f"LogReg: {calculate_auprc(y, lr_probs):.4f}")
logger.info(f"RandomForest: {calculate_auprc(y, rf_probs):.4f}")
logger.info(f"IsolationForest: {calculate_auprc(y, if_probs):.4f}")

# Feature importance
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

lr_coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": log_reg.coef_[0],
    "Importance": np.abs(log_reg.coef_[0])
}).sort_values("Importance", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importances.head(10), ax=axes[0])
axes[0].set_title("Random Forest Importance")
sns.barplot(x="Importance", y="Feature", data=lr_coefficients.head(10), ax=axes[1])
axes[1].set_title("Logistic Regression Importance")
plt.tight_layout()

feature_plot_path = REPORTS_DIR / "figures" / "feature_importance.png"
feature_plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(feature_plot_path, dpi=300)
logger.success(f"Feature importance saved to {feature_plot_path}")