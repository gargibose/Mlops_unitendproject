import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation   
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from dataclasses import dataclass #this is used to create a class with attributes that can be initialized with default values   
from src.utils import save_object #this is used to import the save_object function from the utils.py file
print("Script Started")
@dataclass# this decorator is used to create a class with attributes that can be initialized with default values
class DataIngestionConfig:# this class is used to define the configuration for data ingestion where to save the data
    train_data_path: str = os.path.join('artifacts', 'train.csv')   # this is the path where the train data will be saved
    test_data_path: str = os.path.join('artifacts', 'test.csv') # this is the path where the test data will be saved
    raw_data_path: str = os.path.join('artifacts', 'data.csv') # this is the path where the raw data will be saved
    '''This dataclass defines configuration values as fields
      and automatically creates the setup code, so you can easily reuse those settings throughout your pipeline'''  
class DataIngestion: # this class is used to define the data ingestion process
    def __init__(self): # this is the constructor of the class
        self.ingestion_config = DataIngestionConfig() # this is used to create an instance of the DataIngestionConfig class

    def initiate_data_ingestion(self): # this method is used to initiate the data ingestion process
        logging.info("Entered the data ingestion method or component") # this is used to log the information that the data ingestion process has started
        try:
            df = pd.read_csv('Notebook/Data/titanic.csv') # this is used to read the raw data from the csv file
            logging.info("Read the dataset as dataframe") # this is used to log the information that the data has been read successfully

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # this is used to create the directory where the train data will be saved
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) # this is used to save the raw data to the specified path

            logging.info("Train test split initiated") # this is used to log the information that the train test split process has started
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) # this is used to split the data into train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) # this is used to save the train data to the specified path
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) # this is used to save the test data to the specified path
            logging.info("Ingestion of the data is completed") # this is used to log the information that the data ingestion process has completed

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path) # this is used to return the paths of the train and test data
        except Exception as e:
            raise CustomException(e, sys) # this is used to raise a custom exception if any error occurs during the data ingestion process
'''if __name__ == "__main__":

    obj = DataIngestion()

    train_data_path, test_data_path = (
        obj.initiate_data_ingestion()
    )

    print("Train Path:", train_data_path)
    print("Test Path:", test_data_path)

    data_transformation = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiate_data_transformation(
            train_data_path,
            test_data_path
        )
    )

    print("\nTransformation completed")
    print("Preprocessor Path:", preprocessor_path) #this is used to print the path of the preprocessor object and test if 
    the data transformation process is working fine and pickle file of the preprocessor object is created successfully'''
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print("Train Path:", train_data)
    print("Test Path:", test_data)

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
