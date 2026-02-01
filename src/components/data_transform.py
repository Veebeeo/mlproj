import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_obj

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTrans:
    def __init__(self):
        self.data_trans_config = DataTransConfig()
    
    def get_data_transformer_obj(self):
        #Function reponsible for Data Transformation
        try:
            numerical_columns = ["writing_score","reading_score"]
            categ_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "median")), #handling missing val
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    
                ]
            )

            logging.info("numerical columns standard scaling done")
            logging.info("categorical columns encoding done")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categ_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_trans(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data done")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_obj()

            target_column = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feat_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feat_train_df = train_df[target_column]

            input_feat_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feat_test_df = test_df[target_column]

            logging.info(f"Applying preprocesssing obj on training and testing dataframe")

            input_feat_train_array = preprocessing_obj.fit_transform(input_feat_train_df)
            input_feat_test_array = preprocessing_obj.transform(input_feat_test_df)
            
            train_arr = np.c_[
                input_feat_train_array, np.array(target_feat_train_df)
            ]

            test_arr = np.c_[
                input_feat_test_array, np.array(target_feat_test_df)
            ]

            save_obj(
                file_path = self.data_trans_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_trans_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)


