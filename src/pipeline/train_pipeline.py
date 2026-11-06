from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    print("=" * 50)
    print("STARTING TRAINING PIPELINE")
    print("=" * 50)

    # 1. Data Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # 2. Data Validation
    validation = DataValidation()

    validation_status = validation.initiate_data_validation(
        train_path,
        test_path
    )

    if validation_status:
        print("Data Validation Successful")
    else:
        print("❌ Data Validation Failed")

        raise Exception("Data Validation Failed")

    # 3. Data Transformation
    transformation = DataTransformation()

    train_arr, test_arr, _ = (
        transformation.initiate_data_transformation(
            train_path,
            test_path
        )
    )

    # 4. Model Training
    trainer = ModelTrainer()

    trainer.initiate_model_trainer(
        train_arr,
        test_arr
    )

    print("=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)