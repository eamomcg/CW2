import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

def main():
    print(">>> [DEBUG] Starting Pipeline with Quality Gate...", flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Full path to input data asset")
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    # --- FIX IS HERE: Define the variable ---
    MIN_F1_THRESHOLD = 0.20

    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        # 1. Load Data
        print(f">>> [DEBUG] Loading parquet: {args.data}", flush=True)
        df = pd.read_parquet(args.data)
        
        # 2. EMERGENCY OPTIMIZATION: Reduce Feature Space
        print(">>> [DEBUG] Reducing dimensionality (1000 -> 50 cols)...", flush=True)
        
        target_col = "IsAnomaly"
        if target_col not in df.columns:
            raise ValueError("Target missing")
            
        cols_to_keep = [target_col] + [c for c in df.columns if c != target_col][:50]
        df = df[cols_to_keep]
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
            
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
        print(f">>> [DEBUG] Training...", flush=True)
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            class_weight='balanced',
            max_depth=10,
            random_state=42,
            n_jobs=1
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

        print(f">>> [RESULT] F1 Score: {f1:.4f} (Threshold: {MIN_F1_THRESHOLD})", flush=True)

        # --- THE QUALITY GATE ---
        if f1 < MIN_F1_THRESHOLD:
            error_msg = f"Quality Fail: F1 ({f1:.4f}) < Threshold ({MIN_F1_THRESHOLD}). Deployment Aborted."
            print(f">>> [FAILURE] {error_msg}")
            raise ValueError(error_msg)
        
        print(">>> [SUCCESS] Quality Gate Passed. Proceeding to Registration.", flush=True)

if __name__ == "__main__":
    main()