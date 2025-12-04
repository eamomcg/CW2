import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

def main():
    print(">>> [DEBUG] Starting Speed-Run Script...", flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Full path to input data asset")
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        # 1. Load Data
        print(f">>> [DEBUG] Loading parquet: {args.data}", flush=True)
        df = pd.read_parquet(args.data)
        
        # 2. EMERGENCY OPTIMIZATION: Reduce Feature Space
        # Keep 'IsAnomaly' + the first 50 feature columns only.
        # This reduces RAM usage by 95% and guarantees execution.
        print(">>> [DEBUG] Reducing dimensionality (1000 -> 50 cols) for performance...", flush=True)
        
        target_col = "IsAnomaly"
        if target_col not in df.columns:
            raise ValueError("Target column missing.")
            
        # Select target + first 50 float columns
        cols_to_keep = [target_col] + [c for c in df.columns if c != target_col][:50]
        df = df[cols_to_keep]
        
        # Optimize remaining 50 columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
            
        print(f">>> [DEBUG] New Data Shape: {df.shape}", flush=True)

        X = df.drop(target_col, axis=1)
        y = df[target_col]

        del df
        gc.collect()

        # 3. Split
        print(">>> [DEBUG] Splitting...", flush=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # 4. Train
        print(f">>> [DEBUG] Training...", flush=True)
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            class_weight='balanced',
            max_depth=10,  # Shallow trees for speed
            random_state=42,
            n_jobs=1
        )
        model.fit(X_train, y_train)
        print(">>> [DEBUG] Training Complete.", flush=True)

       # 5. Evaluate
        y_pred = model.predict(X_test)
        
        # Calculate full suite of metrics for Slide 8
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        mlflow.log_metric("val_f1", f1)
        mlflow.log_metric("val_precision", precision)
        mlflow.log_metric("val_recall", recall)

        print(f">>> [RESULT] F1 Score: {f1:.4f} (Threshold: {MIN_F1_THRESHOLD})", flush=True)

        # --- THE QUALITY GATE ---
        # This block ensures we abide by Assessment Criterion 3
        if f1 < MIN_F1_THRESHOLD:
            error_msg = f"Quality Fail: F1 ({f1:.4f}) < Threshold ({MIN_F1_THRESHOLD}). Deployment Aborted."
            print(f">>> [FAILURE] {error_msg}")
            raise ValueError(error_msg)
        
        print(">>> [SUCCESS] Quality Gate Passed. Proceeding to Registration.", flush=True)

if __name__ == "__main__":
    main()