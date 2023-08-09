import os
import sys
from APS.logger import logging
from APS.exception import CustomException
import pandas as pd
from from_root import from_root

from APS.components.data_ingestion import DataIngestion
from APS.components.data_transformation import DataTransformation
from APS.components.model_trainer import ModelTrainer


def start_training_pipeline():
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_arr, test_arr)


'''
if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_arr, test_arr)
'''