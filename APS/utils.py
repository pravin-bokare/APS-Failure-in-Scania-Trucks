import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from APS.exception import CustomException
from APS.logger import logging
import yaml


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report_test_data = {}
        report_train_data = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            logging.info(f'Training started for {model}')
            model.fit(X_train, y_train)
            logging.info(f'Training Completed for {model}')

            # Predict Testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Get R2 scores for train and test data
            # train_model_score = r2_score(y_train,y_train_pred)

            train_model_score = f1_score(y_train, y_train_pred)
            test_model_score = f1_score(y_test, y_test_pred)

            report_test_data[list(models.keys())[i]] = test_model_score
            report_train_data[list(models.keys())[i]] = train_model_score

            if test_model_score < 0.7:
                logging.info('Model Score is Not Good')

        return report_test_data, report_train_data

    except Exception as e:
        logging.info('Exception occurred during model training')
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occurred in load_object function utils')
        raise CustomException(e, sys)
