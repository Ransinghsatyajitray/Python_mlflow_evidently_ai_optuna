import pandas as pd
import numpy as np
import os
import sys
from src.irisclassification.logger import logging
from src.irisclassification.exception import customexception
from dataclasses import dataclass
from src.irisclassification.utils.utils import save_object
from src.irisclassification.utils.utils import evaluate_model

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# @dataclass is a new feature, if we place it no need to create __init__ method
@dataclass 
class ModelTrainerConfig: 
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_array, test_array):     # this will be taking input from data transformation array outputs
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            
            # the classification models we will be using
            models = {
            'SVC': SVC(),
            'RandomForestClassifier': RandomForestClassifier()
            }
            
            param_grids = {
            'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
            'RandomForestClassifier': {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10], 'random_state': [30]}
}
            
            # evaluate model is from utils
            model_report = evaluate_model(models, param_grids, X_train, y_train, X_test, y_test)

            # best model and parameters
            best_model_report = model_report["best_clf_model"]

            logging.info(f'Best Model Found , Model Name : {best_model_report}') # log the classification scores

            # from the utils.py (utility)
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj = best_model_report                   
            )

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e, sys)

        
    