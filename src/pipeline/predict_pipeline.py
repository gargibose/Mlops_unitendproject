import sys 
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
class PredictPipeline:# this class is used to define the prediction pipeline
    def __init__(self):
        pass
    def predict(self, features):# this method is used to make predictions on the input features
        try:
            preprocessor_path = 'artifacts/preprocessor.pkl' # this is the path where the preprocessor object is saved
            model_path = 'artifacts/model.pkl' # this is the path where the trained model is saved
            preprocessor = load_object(file_path=preprocessor_path) # this is used to load the preprocessor object from the specified path
            model = load_object(file_path=model_path) # this is used to load the trained model from the specified path
            
            data_scaled = preprocessor.transform(features) # this is used to transform the input features using the preprocessor object
            preds = model.predict(data_scaled) # this is used to make predictions on the transformed features using the trained model
            proba = model.predict_proba(data_scaled)# this is used to get the probabilities of the predictions made by the trained model

            return preds, proba # this is used to return the predictions and probabilities
        except Exception as e:
            raise CustomException(e, sys) # this is used to raise a custom exception if any error occurs during the prediction process

class CustomData:# this class is used to define the custom data for prediction
    def __init__(self, Pclass: int, Sex: str, Age: float, SibSp: int, Parch: int, Fare: float, Embarked: str):
        self.Pclass = Pclass
        self.Sex = Sex
        self.Age = Age
        self.SibSp = SibSp
        self.Parch = Parch
        self.Fare = Fare
        self.Embarked = Embarked

    def get_data_as_dataframe(self):# this method is used to convert the custom data into a dataframe it will return the data as in dataframe
        try:
            custom_data_input_dict = {
                'Pclass': [self.Pclass],
                'Sex': [self.Sex],
                'Age': [self.Age],
                'SibSp': [self.SibSp],
                'Parch': [self.Parch],
                'Fare': [self.Fare],
                'Embarked': [self.Embarked]
            }
            return pd.DataFrame(custom_data_input_dict)# this is used to convert the custom data into a dataframe
        except Exception as e:
            raise CustomException(e, sys)
        
