from APS.exception import CustomException
from APS.logger import logging
import pandas as pd
from datetime import datetime
import os
from APS.utils import load_object
from from_root import from_root
import sys
import numpy as np

PREDICTION_DIR = os.path.join(from_root(), 'predictions')
PREDICTION_FILE_NAME = f'{datetime.now().strftime("%d%Y__%H%M%S")}'


def start_batch_predictions(input_file_path):
    try:
        logging.info(f'Reading file : {input_file_path}')
        df = pd.read_csv(input_file_path)
        df.replace({"na": np.NAN}, inplace=True)

        logging.info(f'Loading Transformer Object to daata transformation')
        transformer = load_object(os.path.join(from_root(), 'artifacts', 'preprocessor.pkl'))

        input_feature_name = list(transformer.feature_names_in_)

        input_arr = transformer.transform(df[input_feature_name])

        logging.info(f'Loading model object to prediction')
        model = load_object(os.path.join(from_root(), 'artifacts', 'model.pkl'))
        #print(input_arr)
        prediction = model.predict(input_arr)

        logging.info(f'Loading Label Encoder Object to Encoding')
        target_encoder = load_object(os.path.join(from_root(), 'artifacts', 'label_encoder.pkl'))

        cat_prediction = target_encoder.inverse_transform(prediction)

        df['predictions'] = prediction
        df['cat_predictions'] = cat_prediction

        prediction_file_name = os.path.basename(input_file_path).replace(".csv",
                                                                         f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR, prediction_file_name)
        #print(prediction_file_path)
        df.to_csv(prediction_file_path, index=False, header=True)
        return prediction_file_path
    except Exception as e:
        raise CustomException(e, sys)

        prediction_file_name = os.path.base(input_file_path).replace('.csv',f'{datetime.now().strftime("%d%Y__%H%M%S")}.csv')

    except Exception as e:
        print(e)
