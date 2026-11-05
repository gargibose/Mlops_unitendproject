import os
import sys
import pandas as pd

from evidently import Report
from evidently.presets import (
    DataDriftPreset,
    DataSummaryPreset
)

from src.exception import CustomException

from src.utils import (
    create_directory,
    classification_report,
    save_html_report
)


class EvidentlyMonitor:

    def __init__(self):

        self.report_dir = "artifacts/reports"

        create_directory(
            self.report_dir
        )

    def generate_classification_report(
        self,
        y_true,
        y_pred
    ):

        try:

            metrics = classification_report(
                y_true,
                y_pred
            )

            report_path = os.path.join(
                self.report_dir,
                "classification_report.html"
            )

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Performance Dashboard</title>

                <style>

                    body {{
                        font-family: Arial, sans-serif;
                        background: #f4f6f9;
                        padding: 30px;
                    }}

                    h1 {{
                        text-align: center;
                        color: #2563eb;
                    }}

                    .cards {{
                        display: grid;
                        grid-template-columns:
                        repeat(auto-fit, minmax(220px, 1fr));
                        gap: 20px;
                        margin-top: 40px;
                    }}

                    .card {{
                        background: white;
                        padding: 25px;
                        border-radius: 15px;
                        box-shadow:
                        0px 4px 12px rgba(0,0,0,0.1);
                        text-align: center;
                    }}

                    .value {{
                        font-size: 32px;
                        font-weight: bold;
                        color: #2563eb;
                    }}

                    .label {{
                        color: #666;
                        margin-top: 8px;
                    }}

                </style>

            </head>

            <body>

                <h1>📊 Titanic Model Performance</h1>

                <div class="cards">

                    <div class="card">
                        <div class="value">
                            {metrics['accuracy']*100:.2f}%
                        </div>
                        <div class="label">
                            Accuracy
                        </div>
                    </div>

                    <div class="card">
                        <div class="value">
                            {metrics['precision']*100:.2f}%
                        </div>
                        <div class="label">
                            Precision
                        </div>
                    </div>

                    <div class="card">
                        <div class="value">
                            {metrics['recall']*100:.2f}%
                        </div>
                        <div class="label">
                            Recall
                        </div>
                    </div>

                    <div class="card">
                        <div class="value">
                            {metrics['f1_score']*100:.2f}%
                        </div>
                        <div class="label">
                            F1 Score
                        </div>
                    </div>

                </div>

            </body>
            </html>
            """

            with open(
                report_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(html_content)

            print(
                f"Classification Report Saved: {report_path}"
            )

            return report_path

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    def generate_data_drift_report(
        self,
        train_df,
        test_df
    ):

        try:

            report = Report([
                DataDriftPreset()
            ])

            snapshot = report.run(
                reference_data=train_df,
                current_data=test_df
            )

            report_path = os.path.join(
                self.report_dir,
                "data_drift_report.html"
            )

            save_html_report(
                snapshot,
                report_path
            )

            print(
                f"Drift Report Saved: {report_path}"
            )

            return report_path

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    def generate_data_summary_report(
        self,
        dataframe
    ):

        try:

            report = Report([
                DataSummaryPreset()
            ])

            snapshot = report.run(
                current_data=dataframe,
                reference_data=None
            )

            report_path = os.path.join(
                self.report_dir,
                "data_summary_report.html"
            )

            save_html_report(
                snapshot,
                report_path
            )

            print(
                f"Summary Report Saved: {report_path}"
            )

            return report_path

        except Exception as e:

            raise CustomException(
                e,
                sys
            )