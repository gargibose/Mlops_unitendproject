###read the dataset from utils code
import pandas as pd
import os
import dill 
import sys
from src.exception import CustomException
from sklearn.metrics import f1_score 
def save_object(file_path, obj):
    '''This function is responsible for saving the object to the specified file path'''
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
def evaluate_models(X_train, y_train, X_test, y_test, models):
    '''This function is responsible for evaluating the models and returning the model report'''
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train) # this is used to fit the model on the training data
            y_test_pred = model.predict(X_test) # this is used to predict the target variable on the test data
            test_model_score = f1_score(y_test, y_test_pred) # this is used to calculate the f1 score of the model on the test data
            report[list(models.keys())[i]] = test_model_score # this is used to store the f1 score of the model in the report dictionary
        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    '''This function is responsible for loading the object from the specified file path'''
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)    