import os
import sys
from APS.logger import logging
from APS.exception import CustomException
import pandas as pd
from scipy.stats import ks_2samp
from typing import Optional
import numpy as np
from APS import utils
from dataclasses import dataclass
from from_root import from_root


@dataclass
class DataValidationConfig:
    data_valiation_report_path = os.path.join(from_root(), 'artifacts', 'report.yaml')
    validation_error = dict()


    def drop_missing_values_columns(self, df):
        try:
            threshold = self.missing_threshold
            null_report = df.isna().sum() / df.shape[0]
            # selecting column name which contains null
            logging.info(f"selecting column name which contains null above to {threshold}")
            drop_column_names = null_report[null_report > threshold].index

            logging.info(f"Columns to drop: {list(drop_column_names)}")
            df.drop(list(drop_column_names), axis=1, inplace=True)

            # return None no columns left
            if len(df.columns) == 0:
                return None
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def is_required_columns_exists(self, base_df, current_df, report_key_name):
        try:

            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Column: [{base_column} is not available.]")
                    missing_columns.append(base_column)

            if len(missing_columns) > 0:
                validation_error[report_key_name] = missing_columns
                return False
            return True
        except Exception as e:
            raise SensorException(e, sys)

    def data_drift(self, base_df, current_df, report_key_name):
        try:
            drift_report = dict()

            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data, current_data = base_df[base_column], current_df[base_column]
                # Null hypothesis is that both column data drawn from same distrubtion

                logging.info(f"Hypothesis {base_column}: {base_data.dtype}, {current_data.dtype} ")
                same_distribution = ks_2samp(base_data, current_data)

                if same_distribution.pvalue > 0.05:
                    # We are accepting null hypothesis
                    drift_report[base_column] = {
                        "pvalues": float(same_distribution.pvalue),
                        "same_distribution": True
                    }
                else:
                    drift_report[base_column] = {
                        "pvalues": float(same_distribution.pvalue),
                        "same_distribution": False
                    }
                    # different distribution

            self.validation_error[report_key_name] = drift_report
        except Exception as e:
            raise SensorException(e, sys)

@dataclass
class DataValidation:
    data_validation_config = DataValidationConfig(),
    base_data_path = os.path.join(from_root(), 'data', 'aps_failure_training_set.csv')
    train_data_path = os.path.join(from_root(),'artifacts', 'train.csv')
    test_data_path = os.path.join(from_root(), 'artifacts', 'test.csv')
    validation_error = dict()

    def initiate_data_validation(self):
        try:
            logging.info("Reading base dataframe")
            base_df = pd.read_csv(self.base_data_path)
            base_df.replace({"na": np.NAN}, inplace=True)
            logging.info("Replace na value in base df")
            # base_df has na as null
            logging.info("Drop null values columns from base df")
            base_df = self.data_validation_config.drop_missing_values_columns(df=base_df, report_key_name="missing_values_within_base_dataset")

            logging.info("Reading train dataframe")
            train_df = pd.read_csv(self.train_data_path)
            logging.info("Reading test dataframe")
            test_df = pd.read_csv(self.test_data_path)

            logging.info("Drop null values columns from train df")
            train_df = self.data_validation_config.drop_missing_values_columns(df=train_df, report_key_name="missing_values_within_train_dataset")
            logging.info("Drop null values columns from test df")
            test_df = self.data_validation_config.drop_missing_values_columns(df=test_df, report_key_name="missing_values_within_test_dataset")

            exclude_columns = [TARGET_COLUMN]
            base_df = utils.convert_columns_float(df=base_df, exclude_columns=exclude_columns)
            train_df = utils.convert_columns_float(df=train_df, exclude_columns=exclude_columns)
            test_df = utils.convert_columns_float(df=test_df, exclude_columns=exclude_columns)

            logging.info("Is all required columns present in train df")
            train_df_columns_status = self.data_validation_config.is_required_columns_exists(base_df=base_df, current_df=train_df)
            logging.info("Is all required columns present in test df")
            test_df_columns_status = self.data_validation_config.is_required_columns_exists(base_df=base_df, current_df=test_df)

            if train_df_columns_status:
                logging.info("As all column are available in train df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=train_df)
            if test_df_columns_status:
                logging.info("As all column are available in test df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=test_df)

            # write the report
            logging.info("Write report in yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
                                  data=self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path)
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)
