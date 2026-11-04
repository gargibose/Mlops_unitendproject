import os 
import sys
#models libraries from sklearn, xgboost and catboost
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
#importing the model evaluation metrics from sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl') # this is the path where the trained model will be saved   
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() # this is used to create an instance of the ModelTrainerConfig class
    def initiate_model_trainer(self, train_array, test_array): # this method is used to initiate the model training process
        try:
            logging.info("Splitting training and test input data") # this is used to log the information that the training and test data is being split
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], # this is used to get all the columns except the last column as the input features for training
                train_array[:, -1], # this is used to get the last column as the target variable for training
                test_array[:, :-1], # this is used to get all the columns except the last column as the input features for testing
                test_array[:, -1] # this is used to get the last column as the target variable for testing
            )
            models = { # this is a dictionary that contains the models that will be trained
                "Logistic Regression": LogisticRegression(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Extra Trees": ExtraTreesClassifier(),
                "Gradient Boosting": GradientBoostingClassifier()
            }
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models) # this is a dictionary that will be used to store the model evaluation metrics for each model
            #to get the best model score from the model report, will store the best model to the based on the f1 score as this is a classification problem and f1 score is a good metric to evaluate the model performance
            best_model_score = max(sorted(model_report.values())) # this is used to get the best model score from the model report
            best_model_name = list(model_report.keys())[ # this is used to get the best model name from the model report
                list(model_report.values()).index(best_model_score) # this is used to get the index of the best model score from the model report
            ]
            best_model = models[best_model_name] # this is used to get the best model from the models dictionary
            if best_model_score < 0.6: # this is used to check if the best model score is less than 0.6
                raise CustomException("No best model found") # this is used to raise a custom exception if no best model is found
            logging.info(f"Best model found on both training and testing dataset: {best_model_name} with accuracy score: {best_model_score}") # this is used to log the information that the best model has been found
            save_object( # this is used to save the best model to the specified path
                file_path=self.model_trainer_config.trained_model_file_path, # this is the path where the trained model will be saved
                obj=best_model # this is the best model that will be saved
            )
            predicted = best_model.predict(X_test) # this is used to predict the target variable for the test data using the best model
            f1_scr = f1_score(y_test, predicted) # this is used to calculate the f1 score for the test data using the best model
            return f1_scr # this is used to return the f1 score for the test data      
        except Exception as e:
            raise CustomException(e, sys) # this is used to raise a custom exception if any error occurs during the model training process  
        