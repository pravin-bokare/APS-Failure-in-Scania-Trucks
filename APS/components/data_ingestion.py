import os
import sys
from APS.logger import logging
from APS.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from APS.logger import logging
from from_root import from_root



@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join(from_root(), 'artifacts')
    test_data_path = os.path.join(from_root(), 'artifacts')
    raw_data_path = os.path.join(from_root(), 'artifacts')


class DataIngestionconfig:
    df_path = os.path.join(from_root(), 'data', 'aps_failure_training_set.csv')
    train_data_path:str=os.path.join(from_root(),'artifacts', 'train.csv')
    test_data_path:str=os.path.join(from_root(), 'artifacts', 'test.csv')
    raw_data_path:str=os.path.join(from_root(), 'artifacts', 'raw.csv')


## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            df = pd.read_csv(self.ingestion_config.df_path)
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Train test split')
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of Data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e, sys)


'''
if __name__=='__main__':
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()

'''