import sys
import os
from APS.exception import CustomException
from APS.logger import logging
from APS.utils import load_object
import pandas as pd
from from_root import from_root
import sys
from APS.pipeline.batch_prediction import start_batch_predictions
file_path = os.path.join(from_root(),'data', 'aps_failure_test_set_1.csv')


if __name__ == '__main__':
    try:
        logging.info(f'Predictions Started')
        output_file = start_batch_predictions(input_file_path=file_path)
        print(output_file)
    except Exception as e:
        logging.info('Exception occurred at Model Training')
        raise CustomException(e, sys)