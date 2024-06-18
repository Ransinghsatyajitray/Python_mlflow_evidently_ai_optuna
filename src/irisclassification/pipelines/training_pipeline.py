from src.irisclassification.components.data_ingestion import DataIngestion
from src.irisclassification.components.data_transformation import DataTransformation
from src.irisclassification.components.model_trainer import ModelTrainer

import os
import sys
from src.irisclassification.logger import logging
from src.irisclassification.exception import customexception

obj = DataIngestion()
train_data_path, test_data_path = obj.initiate_data_ingestion()

data_transformation = DataTransformation()
train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

model_trainer_obj = ModelTrainer()
model_trainer_obj.initate_model_training(train_arr, test_arr)