import os
import sys
import pandas as pd

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataValidationConfig:
    validation_status_file = os.path.join(
        "artifacts",
        "validation_status.txt"
    )


class DataValidation:

    def __init__(self):
        self.validation_config = (
            DataValidationConfig()
        )

    def initiate_data_validation(
        self,
        train_path,
        test_path
    ):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            validation_status = True

            required_columns = [
                "Survived",
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked"
            ]

            for col in required_columns:

                if col not in train_df.columns:
                    validation_status = False

                if col not in test_df.columns:
                    validation_status = False

            os.makedirs(
                "artifacts",
                exist_ok=True
            )

            with open(
                self.validation_config.validation_status_file,
                "w"
            ) as f:

                f.write(
                    f"Validation Status : {validation_status}"
                )

            logging.info(
                f"Validation Status : {validation_status}"
            )

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)