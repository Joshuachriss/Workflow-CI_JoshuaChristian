import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc
)

parser = argparse.ArgumentParser(description='Titanic Survival Prediction - MLflow CI')
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=10,
                    help='Max depth, 0 = None')
parser.add_argument('--min_samples_split', type=int, default=2)
parser.add_argument('--min_samples_leaf', type=int, default=1)
parser.add_argument('--max_features', type=str, default='sqrt')
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()

MAX_DEPTH = None if args.max_depth == 0 else args.max_depth

DATA_PATH = "titanic_preprocessing/titanic_preprocessing.csv"
EXPERIMENT_NAME = "titanic-ci-pipeline"
ARTIFACT_DIR = "mlproject_artifacts"

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return train_test_split(X, y, test_size=args.test_size,
                            random_state=args.random_state, stratify=y)


def plot_confusion_matrix(y_test, y_pred, output_dir):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Tidak Selamat', 'Selamat'],
                yticklabels=['Tidak Selamat', 'Selamat'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix - Titanic')
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=100)
    plt.close()
    return path


def plot_roc_curve(y_test, y_proba, output_dir):
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'AUC = {roc_auc_val:.4f}')
    ax.plot([0, 1], [0, 1], 'navy', lw=1.5, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(path, dpi=100)
    plt.close()
    return path


def plot_feature_importance(model, feature_names, output_dir):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([feature_names[i] for i in idx], importances[idx], color='steelblue')
    ax.set_title('Feature Importance - Titanic')
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(path, dpi=100)
    plt.close()
    return path


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()

    print(f"Training: n_estimators={args.n_estimators}, max_depth={MAX_DEPTH}, "
          f"min_samples_split={args.min_samples_split}")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=MAX_DEPTH,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    # Buat artefak
    cm_path = plot_confusion_matrix(y_test, y_pred, ARTIFACT_DIR)
    roc_path = plot_roc_curve(y_test, y_proba, ARTIFACT_DIR)
    fi_path = plot_feature_importance(model, X_train.columns.tolist(), ARTIFACT_DIR)

    report = classification_report(y_test, y_pred,
                target_names=['Tidak Selamat', 'Selamat'], output_dict=True)
    cr_path = os.path.join(ARTIFACT_DIR, "classification_report.json")
    with open(cr_path, 'w') as f:
        json.dump(report, f, indent=4)

    model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
    joblib.dump(model, model_path)

    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment("titanic-ci-pipeline")
        
    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("max_features", args.max_features)
        mlflow.log_param("test_size", args.test_size)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(fi_path)
        mlflow.log_artifact(cr_path)
        mlflow.log_artifact(model_path)

        print(f"\n=== HASIL ===")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")
        print(f"CV Mean  : {cv_scores.mean():.4f}")
        print(f"Run ID   : {run.info.run_id}")

        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)


if __name__ == "__main__":
    main()
