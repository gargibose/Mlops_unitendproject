import os
import sys

# MLflow libraries
import mlflow
import mlflow.sklearn

# Models libraries from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

# Importing model evaluation metrics from sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from dataclasses import dataclass

from src.Monitoring.Evidently_report import EvidentlyMonitor
from src.exception import CustomException
from src.logger import logging

from src.utils import (
    save_object,
    evaluate_models
)


@dataclass
class ModelTrainerConfig:
    # this is the path where the trained model
    # will be saved
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        # this is used to create an instance
        # of the ModelTrainerConfig class
        self.model_trainer_config = (
            ModelTrainerConfig()
        )

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Splitting training and testing data"
            )

            # this is used to split train and test
            # arrays into input features and target variable

            X_train, y_train, X_test, y_test = (

                train_array[:, :-1],  # training features
                train_array[:, -1],   # training target

                test_array[:, :-1],   # testing features
                test_array[:, -1]     # testing target
            )

            # this is a dictionary containing all
            # machine learning models to evaluate

            models = {

                "Logistic Regression":
                LogisticRegression(),

                "K-Nearest Neighbors":
                KNeighborsClassifier(),

                "Decision Tree":
                DecisionTreeClassifier(),

                "Random Forest":
                RandomForestClassifier(),

                "AdaBoost":
                AdaBoostClassifier(),

                "Extra Trees":
                ExtraTreesClassifier(),

                "Gradient Boosting":
                GradientBoostingClassifier()

            }

            logging.info(
                "Model evaluation started"
            )

            # this function trains all models and
            # returns their F1 scores

            model_report = evaluate_models(

                X_train=X_train,
                y_train=y_train,

                X_test=X_test,
                y_test=y_test,

                models=models

            )

            # this is used to get the best score
            # among all trained models

            best_model_score = max(
                sorted(
                    model_report.values()
                )
            )

            # this is used to identify the
            # best model name

            best_model_name = list(
                model_report.keys()
            )[
                list(
                    model_report.values()
                ).index(
                    best_model_score
                )
            ]

            # this is used to retrieve the
            # best model object

            best_model = models[
                best_model_name
            ]

            # this is used to make sure
            # the model performs reasonably well

            if best_model_score < 0.60:

                raise CustomException(
                    "No best model found",
                    sys
                )

            logging.info(
                f"Best model found : "
                f"{best_model_name}"
            )
            print("\n🏆 BEST MODEL")
            print("=" * 50)
            print(f"Model Name : {best_model_name}")
            print(f"F1 Score   : {best_model_score:.4f}")
            print("=" * 50)
            logging.info(
                f"Best model F1 score : "
                f"{best_model_score}"
            )


            # this is used to save the
            # best model as model.pkl

            save_object(

                file_path=
                self.model_trainer_config
                .trained_model_file_path,

                obj=best_model

            )

            # this is used to make predictions
            # on the test dataset

            predicted = best_model.predict(
                X_test
            )

            # ==================================================
            # Model Evaluation Metrics
            # ==================================================

            # this is used to calculate
            # the accuracy score

            accuracy = accuracy_score(
                y_test,
                predicted
            )

            # this is used to calculate
            # the precision score

            precision = precision_score(
                y_test,
                predicted
            )

            # this is used to calculate
            # the recall score

            recall = recall_score(
                y_test,
                predicted
            )

            # this is used to calculate
            # the F1 score

            f1_scr = f1_score(
                y_test,
                predicted
            )
            print("\n📈 FINAL METRICS")
            print("=" * 50)
            print(f"Accuracy  : {accuracy:.4f}")
            print(f"Precision : {precision:.4f}")
            print(f"Recall    : {recall:.4f}")
            print(f"F1 Score  : {f1_scr:.4f}")
            print("=" * 50)
            # ==================================================
            # MLFLOW EXPERIMENT TRACKING
            # ==================================================

            # this is used to create an experiment
            # inside MLflow

            mlflow.set_experiment(
                "Titanic_Survival_Prediction"
            )

            # this is used to start an MLflow run

            with mlflow.start_run():

                # log model information

                mlflow.log_param(
                    "best_model_name",
                    best_model_name
                )

                # log metrics

                mlflow.log_metric(
                    "accuracy",
                    accuracy
                )

                mlflow.log_metric(
                    "precision",
                    precision
                )

                mlflow.log_metric(
                    "recall",
                    recall
                )

                mlflow.log_metric(
                    "f1_score",
                    f1_scr
                )

                mlflow.log_metric(
                    "best_model_score",
                    best_model_score
                )

                # this is used to store
                # the trained model inside MLflow

                '''mlflow.sklearn.log_model(

                    sk_model=best_model,

                    name=
                    "best_model"

                )'''
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    name="best_model",
                    skops_trusted_types=[
                         "sklearn.tree._tree.Tree"
                    ]
                )

                logging.info(
                    "MLflow experiment "
                    "logged successfully"
                )

            # ==================================================
            # EVIDENTLY REPORT GENERATION
            # ==================================================

            # this is used to generate a
            # classification monitoring report

            monitor = EvidentlyMonitor()

            monitor.generate_classification_report(

                y_true=y_test,
                y_pred=predicted

            )

            logging.info(
                f"Classification report "
                f"generated for "
                f"{best_model_name}"
            )

            # this is used to return the
            # final F1 s

            return f1_scr

        except Exception as e:

            raise CustomException(
                e,
                sys
            )