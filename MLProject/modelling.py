"""
modelling.py — MLflow Project (Workflow CI)
============================================================
Script pelatihan model yang digunakan dalam MLflow Project
untuk re-training otomatis via GitHub Actions CI.

Dataset   : Diabetes Prediction Dataset (preprocessed)
Model     : Random Forest Classifier + GridSearchCV
Tracking  : MLflow via DagsHub (online)
Author    : Dwi Cahyani Desri
Version   : 2.0.0 (CI version)
"""

import os
import json
import logging
import warnings
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    log_loss, matthews_corrcoef,
)
from sklearn.utils import estimator_html_repr

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# KONFIGURASI — dibaca dari environment variable (GitHub Secrets)
# ============================================================
DAGSHUB_USERNAME  = os.environ.get("DAGSHUB_USERNAME", "dwicahyanidesri")
DAGSHUB_REPO_NAME = os.environ.get("DAGSHUB_REPO_NAME", "Membangun_model_Dwi-Cahyani-Desri")
EXPERIMENT_NAME   = "Diabetes Prediction - CI"

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH  = os.path.join(SCRIPT_DIR, "diabetes_preprocessing", "diabetes_train_preprocessed.csv")
TEST_PATH   = os.path.join(SCRIPT_DIR, "diabetes_preprocessing", "diabetes_test_preprocessed.csv")


# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    logger.info(f"Memuat data train : {TRAIN_PATH}")
    logger.info(f"Memuat data test  : {TEST_PATH}")

    df_train = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)

    X_train = df_train.drop(columns=["diabetes"])
    y_train = df_train["diabetes"]
    X_test  = df_test.drop(columns=["diabetes"])
    y_test  = df_test["diabetes"]

    logger.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ============================================================
# TRAINING + LOGGING
# ============================================================
def train():
    # Set tracking URI langsung via env var (CI-friendly, tanpa dagshub.init interaktif)
    tracking_uri = (
        f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(disable=True)

    X_train, X_test, y_train, y_test = load_data()

    # Hyperparameter tuning
    param_grid = {
        "n_estimators"    : [100, 200],
        "max_depth"       : [8, 10, 15],
        "min_samples_split": [2, 5],
        "class_weight"    : ["balanced"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=1,
    )
    logger.info("Memulai GridSearchCV...")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    logger.info(f"Best params: {grid_search.best_params_}")

    with mlflow.start_run(run_name="RandomForest_CI") as run:
        logger.info(f"Run ID: {run.info.run_id}")

        # Log params
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("random_state", 42)

        # Prediksi & metrik
        y_pred        = best_model.predict(X_test)
        y_pred_proba  = best_model.predict_proba(X_test)[:, 1]
        y_train_pred  = best_model.predict(X_train)

        metrics = {
            "accuracy"          : round(accuracy_score(y_test, y_pred), 6),
            "precision"         : round(precision_score(y_test, y_pred, zero_division=0), 6),
            "recall"            : round(recall_score(y_test, y_pred, zero_division=0), 6),
            "f1_score"          : round(f1_score(y_test, y_pred, zero_division=0), 6),
            "roc_auc"           : round(roc_auc_score(y_test, y_pred_proba), 6),
            "log_loss"          : round(log_loss(y_test, y_pred_proba), 6),
            "matthews_corrcoef" : round(matthews_corrcoef(y_test, y_pred), 6),
            "cv_best_f1_score"  : round(grid_search.best_score_, 6),
            "training_accuracy" : round(accuracy_score(y_train, y_train_pred), 6),
        }
        mlflow.log_metrics(metrics)

        logger.info("=" * 50)
        for k, v in metrics.items():
            logger.info(f"  {k:<22}: {v}")
        logger.info("=" * 50)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=["Tidak Diabetes", "Diabetes"])
            fig, ax = plt.subplots(figsize=(7, 6))
            disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
            ax.set_title("Confusion Matrix — CI Run", fontsize=13, fontweight="bold")
            plt.tight_layout()
            cm_path = os.path.join(tmpdir, "training_confusion_matrix.png")
            plt.savefig(cm_path, dpi=150, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(cm_path)

            # 2. metric_info.json
            info = {
                "model": "RandomForestClassifier",
                "tuning": "GridSearchCV",
                "best_params": grid_search.best_params_,
                "metrics": metrics,
            }
            mj_path = os.path.join(tmpdir, "metric_info.json")
            with open(mj_path, "w") as f:
                json.dump(info, f, indent=4)
            mlflow.log_artifact(mj_path)

            # 3. estimator.html
            eh_path = os.path.join(tmpdir, "estimator.html")
            with open(eh_path, "w") as f:
                f.write(estimator_html_repr(best_model))
            mlflow.log_artifact(eh_path)

            # 4. classification_report.txt
            report_path = os.path.join(tmpdir, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write(classification_report(y_test, y_pred,
                        target_names=["Tidak Diabetes", "Diabetes"]))
            mlflow.log_artifact(report_path)

            # 5. feature_importance.png
            importances = best_model.feature_importances_
            feature_names = X_train.columns.tolist()
            sorted_idx = importances.argsort()[::-1]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(feature_names)), importances[sorted_idx])
            ax.set_xticks(range(len(feature_names)))
            ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=35, ha="right")
            ax.set_title("Feature Importance — CI Run", fontsize=13, fontweight="bold")
            plt.tight_layout()
            fi_path = os.path.join(tmpdir, "feature_importance.png")
            plt.savefig(fi_path, dpi=150, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(fi_path)

        # Model
        signature = infer_signature(X_train, y_pred_proba)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name="DiabetesPredictionModel_CI",
        )

        logger.info(f"Selesai. Run ID: {run.info.run_id}")
        print(f"MLFLOW_RUN_ID={run.info.run_id}")


if __name__ == "__main__":
    train()