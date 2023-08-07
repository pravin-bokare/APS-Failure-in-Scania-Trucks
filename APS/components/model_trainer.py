# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from APS.exception import CustomException
from APS.logger import logging

from APS.utils import save_object
from APS.utils import evaluate_model

from dataclasses import dataclass
import sys
import os
from from_root import from_root


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(from_root(), 'artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train = train_array[:, :-1], train_array[:,-1]
            X_test, y_test = test_array[:, :-1], test_array[:,-1]
            logging.info('Splitting Completed')

            #print(f'X_train shape {X_train.shape}')
            #print(f'y_train shape {y_train.shape}')
            #print(f'X_test shape {X_test.shape}')
            #print(f'y_test shape {y_test.shape}')

            models = {
                'LogisticRegression': LogisticRegression(max_iter=2000),
                'SVM': SVC(),
                'XgboostClassifier': XGBClassifier(objective='binary:logistic', tree_method='gpu_hist'),
            }

            test_model_report, train_model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(f'Test Model Report {test_model_report}')
            print('\n====================================================================================\n')
            logging.info(f'Test Model Report : {test_model_report}')

            print(f'Train Model Reort {train_model_report}')
            print('\n====================================================================================\n')
            logging.info(f'Train Model Report : {train_model_report}')

            # To get best model score from dictionary
            best_model_score = max(sorted(test_model_report.values()))

            best_model_name = list(test_model_report.keys())[
                list(test_model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , f1 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , f1 Score : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )




        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)