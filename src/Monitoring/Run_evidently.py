import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.Monitoring.Evidently_report import (
    EvidentlyMonitor
)

print("=" * 50)
print("Starting Evidently Monitoring")
print("=" * 50)

try:

    monitor = EvidentlyMonitor()

    train_path = "artifacts/train.csv"
    test_path = "artifacts/test.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"{train_path} not found"
        )

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"{test_path} not found"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # ====================================================
    # Classification Report
    # ====================================================

    target_column = "Survived"

    if target_column in train_df.columns:

        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]

        # Remove non-numeric columns for quick monitoring
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestClassifier(
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        classification_path = (
            monitor.generate_classification_report(
                y_true=y_test,
                y_pred=y_pred
            )
        )

        print(
            f"Classification Report Generated: "
            f"{classification_path}"
        )

    else:

        print(
            f"WARNING: '{target_column}' column not found."
        )

    # ====================================================
    # Data Summary Report
    # ====================================================

    summary_path = (
        monitor.generate_data_summary_report(
            train_df
        )
    )

    print(
        f"Summary Report Generated: "
        f"{summary_path}"
    )

    # ====================================================
    # Data Drift Report
    # ====================================================

    drift_path = (
        monitor.generate_data_drift_report(
            train_df,
            test_df
        )
    )

    print(
        f"Drift Report Generated: "
        f"{drift_path}"
    )

    print("\nReports Generated Successfully\n")

except Exception as e:

    print("\nERROR OCCURRED\n")
    print(str(e))