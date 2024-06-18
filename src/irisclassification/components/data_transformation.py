import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.irisclassification.exception import customexception
from src.irisclassification.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.irisclassification.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl') # we can save the preprocessing steps, training data preprocessing steps we can use on prediction data also


class DataTransformation:
    def __init__(self):
        # creating the object of the configuration
        self.data_transformation_config = DataTransformationConfig()

        
    
    def get_data_transformation(self,train_path):
        
        try:
            logging.info('Data Transformation initiated')
            
            logging.info('Pipeline Initiated')
            
            # Defining the numeric columns
            numerical_col = list(pd.read_csv(train_path).drop(["Species"], axis=1).select_dtypes(exclude="object").columns())
            
            logging.info('Pipeline initiated')
            
            ## Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
                ]
            )
            
            # creating the transformation object
            preprocessor = ColumnTransformer([
            ('num_pipeline', num_pipeline, numerical_cols)
            ])
            
            # this is used in the initialize data transformation for creating the preprocessing object
            return preprocessor
            
        
        except Exception as e:
            
            logging.info("Exception occured in the initiate_datatransformation")
            raise customexception(e,sys)
            
    
    def initialize_data_transformation(self, train_path, test_path):
        try:
            # the paths are in artifact folder
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("reading train and test data complete")
            # just to see what we are entering in data
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            # we have the preprocessor object now
            preprocessing_obj = self.get_data_transformation(train_path)
            
            # Specifying the target column
            target_column_name = 'Species'
            
            # Segregating the independent and dependent feature for training data
            input_feature_train_df = train_df.drop(columns = target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            # Segregating the independent and dependent feature for testing data
            input_feature_test_df = test_df.drop(columns = target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            
            # using the preprocessing object using the methods get_data_transformation
            
            # transformation for training data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            # transformation for testing data (for validation of model)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            
            # we are converting data into numpy object
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] 
            
            
            # Saving the preprocessing object for future processing on unseen data (saved in the path mentioned in preprocessor obj file path). The pkl file will be saved in artifact folder.
            # the save_object is coming from utils.py file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e, sys)
            
    