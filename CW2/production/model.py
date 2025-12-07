import argparse
import os
import gc
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


# Install extra libs needed for training AND to be captured by MLflow env
# - fastparquet: parquet engine for pandas
# - azureml-ai-monitoring: provides azureml.ai.monitoring.Collector used at inference
os.system("pip install fastparquet azureml-ai-monitoring >/dev/null 2>&1")


def main():
    print(">>> [DEBUG] Starting Pipeline with Quality Gate...", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Full path to input data asset")
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    MIN_F1_THRESHOLD = 0.20

    # Let MLflow track params/metrics; we'll explicitly save the model ourselves
    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run():
        # 1. Load Data
        print(f">>> [DEBUG] Loading parquet: {args.data}", flush=True)
        df = pd.read_parquet(args.data, engine="fastparquet")

        # 2. Reduce Feature Space
        print(">>> [DEBUG] Reducing dimensionality (1000 -> 50 cols)...", flush=True)

        target_col = "IsAnomaly"
        if target_col not in df.columns:
            raise ValueError("Target missing")

        cols_to_keep = [target_col] + [c for c in df.columns if c != target_col][:50]
        df = df[cols_to_keep]

        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")

        print(f">>> [DEBUG] New Data Shape: {df.shape}", flush=True)

        X = df.drop(target_col, axis=1)
        y = df[target_col]

        del df
        gc.collect()

        # 3. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # 4. Train
        print(">>> [DEBUG] Training...", flush=True)
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            class_weight="balanced",
            max_depth=10,
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        print(">>> [DEBUG] Training Complete.", flush=True)

        # 5. Evaluate
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        mlflow.log_metric("val_f1", f1)
        mlflow.log_metric("val_precision", precision)
        mlflow.log_metric("val_recall", recall)

        print(
            f">>> [RESULT] F1 Score: {f1:.4f} (Threshold: {MIN_F1_THRESHOLD})",
            flush=True,
        )

        # 6. Quality Gate
        if f1 < MIN_F1_THRESHOLD:
            error_msg = (
                f"Quality Fail: F1 ({f1:.4f}) < Threshold ({MIN_F1_THRESHOLD}). "
                "Deployment Aborted."
            )
            print(f">>> [FAILURE] {error_msg}")
            raise ValueError(error_msg)

        print(">>> [SUCCESS] Quality Gate Passed. Proceeding to Registration.", flush=True)

        # 7. Save MLflow model into ./outputs/model with explicit pip requirements
        outputs_dir = Path("outputs/model")
        outputs_dir.mkdir(parents=True, exist_ok=True)

        print(f">>> [DEBUG] Saving MLflow model to {outputs_dir} ...", flush=True)
        mlflow.sklearn.save_model(
            sk_model=model,
            path=str(outputs_dir),
            pip_requirements=[
                "scikit-learn==1.0.2",
                "pandas<2.0.0",
                "numpy<2.0.0",
                "mlflow",
                "azureml-mlflow",
                "azureml-ai-monitoring",
                "fastparquet",
            ],
        )
        print(">>> [DEBUG] Model saved for AzureML registration.", flush=True)


if __name__ == "__main__":
    main()
