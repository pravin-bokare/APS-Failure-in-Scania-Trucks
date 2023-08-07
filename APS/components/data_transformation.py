import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from APS.exception import CustomException
from APS.logger import logging
import os
from APS.utils import save_object
from from_root import from_root
from imblearn.combine import SMOTETomek
from APS.components.data_ingestion import DataIngestion


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(from_root(), 'artifacts', 'preprocessor.pkl')
    label_encoder_obj_file_path = os.path.join(from_root(), 'artifacts', 'label_encoder.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    @staticmethod
    def get_data_transformation_object():
        try:
            logging.info('Data Transformation Initiated')
            columns = ['aa_000', 'ab_000', 'ac_000', 'ad_000', 'ae_000', 'af_000', 'ag_000', 'ag_001', 'ag_002',
                       'ag_003', 'ag_004', 'ag_005', 'ag_006', 'ag_007', 'ag_008', 'ag_009', 'ah_000', 'ai_000',
                       'aj_000', 'ak_000', 'al_000', 'am_0', 'an_000', 'ao_000', 'ap_000', 'aq_000', 'ar_000',
                       'as_000', 'at_000', 'au_000', 'av_000', 'ax_000', 'ay_000', 'ay_001', 'ay_002', 'ay_003',
                       'ay_004', 'ay_005', 'ay_006', 'ay_007', 'ay_008', 'ay_009', 'az_000', 'az_001', 'az_002',
                       'az_003', 'az_004', 'az_005', 'az_006', 'az_007', 'az_008', 'az_009', 'ba_000', 'ba_001',
                       'ba_002', 'ba_003', 'ba_004', 'ba_005', 'ba_006', 'ba_007', 'ba_008', 'ba_009', 'bb_000',
                       'bc_000', 'bd_000', 'be_000', 'bf_000', 'bg_000', 'bh_000', 'bi_000', 'bj_000', 'bk_000',
                       'bl_000', 'bm_000', 'bn_000', 'bo_000', 'bp_000', 'bq_000', 'br_000', 'bs_000', 'bt_000',
                       'bu_000', 'bv_000', 'bx_000', 'by_000', 'bz_000', 'ca_000', 'cb_000', 'cc_000', 'cd_000',
                       'ce_000', 'cf_000', 'cg_000', 'ch_000', 'ci_000', 'cj_000', 'ck_000', 'cl_000', 'cm_000',
                       'cn_000', 'cn_001', 'cn_002', 'cn_003', 'cn_004', 'cn_005', 'cn_006', 'cn_007', 'cn_008',
                       'cn_009', 'co_000', 'cp_000', 'cq_000', 'cr_000', 'cs_000', 'cs_001', 'cs_002', 'cs_003',
                       'cs_004', 'cs_005', 'cs_006', 'cs_007', 'cs_008', 'cs_009', 'ct_000', 'cu_000', 'cv_000',
                       'cx_000', 'cy_000', 'cz_000', 'da_000', 'db_000', 'dc_000', 'dd_000', 'de_000', 'df_000',
                       'dg_000', 'dh_000', 'di_000', 'dj_000', 'dk_000', 'dl_000', 'dm_000', 'dn_000', 'do_000',
                       'dp_000', 'dq_000', 'dr_000', 'ds_000', 'dt_000', 'du_000', 'dv_000', 'dx_000', 'dy_000',
                       'dz_000', 'ea_000', 'eb_000', 'ec_00', 'ed_000', 'ee_000', 'ee_001', 'ee_002', 'ee_003',
                       'ee_004', 'ee_005', 'ee_006', 'ee_007', 'ee_008', 'ee_009', 'ef_000', 'eg_000']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('StandardScaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, columns),
                ])
            logging.info('Pipeline Completed')
            return preprocessor

        except Exception as e:
            logging.info('Exception occured in Data Transformation')
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # selecting input feature for train and test dataframe
            TARGET_COLUMN = 'class'

            # selecting input feature for train and test dataframe
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            # selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            input_feature_train_df.replace({"na": np.NAN}, inplace=True)
            input_feature_test_df.replace({"na": np.NAN}, inplace=True)

            logging.info('obtaining preprocessing object')
            preprocessor_obj = self.get_data_transformation_object()

            #print(f'Input Feature Test df shape {input_feature_test_df.shape}')
            #print(f'Target Feature Test df shape {target_feature_test_df.shape}')


            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing on train and test data completed successfully')

            smt = SMOTETomek(sampling_strategy='minority')
            logging.info(f'Before resampling training set input {input_feature_train_arr.shape} & target {target_feature_train_arr.shape}')
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, np.array(target_feature_train_arr))
            logging.info(f'After resampling training set input {input_feature_train_arr.shape} & target {target_feature_train_arr.shape}')

            logging.info(f'Before resampling testing set input {input_feature_test_arr.shape} & target {target_feature_test_arr.shape}')
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr,np.array(target_feature_test_arr))
            logging.info(f'After resampling testing set input {input_feature_test_arr.shape} & target {target_feature_test_arr.shape}')

            #print(preprocessor_obj.feature_names_in_)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )
            logging.info('Preprocessor pickle file saved')

            save_object(
                file_path=self.data_transformation_config.label_encoder_obj_file_path,
                obj=label_encoder)
            logging.info('Label Encoder pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.label_encoder_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e, sys)


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

