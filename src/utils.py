###read the dataset from utils code
import pandas as pd
import os
import dill 
import sys
from src.exception import CustomException
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def save_object(file_path, obj):
    '''This function is responsible for saving the object to the specified file path'''
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
def evaluate_models(X_train, y_train, X_test, y_test, models):

    try:

        report = {}

        print("\n📊 MODEL COMPARISON")
        print("=" * 60)

        for model_name, model in models.items():

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = f1_score(
                y_test,
                y_test_pred
            )

            print(
                f"{model_name:<25} "
                f"F1 Score: {test_model_score:.4f}"
            )

            report[model_name] = (
                test_model_score
            )

        print("=" * 60)

        return report

    except Exception as e:

        raise CustomException(
            e,
            sys
        )

def load_object(file_path):
    '''This function is responsible for loading the object from the specified file path'''
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)   
def classification_report(y_true, y_pred):
    '''This function is responsible for generating the classification report'''
    try:
        metrics = {
            "accuracy":round(accuracy_score(y_true, y_pred), 4),
            "precision":round(precision_score(y_true, y_pred), 4),  
            "recall":round(recall_score(y_true, y_pred), 4),
            "f1_score":round(f1_score(y_true, y_pred), 4)   

        }
        return metrics
    except Exception as e:
        raise CustomException(e, sys)    
def create_directory(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
    except Exception as e:
        raise CustomException(e, sys)


def save_html_report(report_object, file_path):
    try:
        report_object.save_html(file_path)
        print(f"Report saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)
    

