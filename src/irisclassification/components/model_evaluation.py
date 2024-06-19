import os
import sys
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score
)
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from src.irisclassification.utils.utils import load_object
import pickle


class ModelEvaluation:
    def __init__(self):
        pass

    
    def eval_metrics(self,actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')
        recall = recall_score(actual, pred, average='weighted')
        return accuracy, precision, f1, recall


    def initiate_model_evaluation(self,train_array, test_array):
        try:
            X_test, y_test=(test_array[:,:-1], test_array[:,-1])
            model_path=os.path.join("artifacts","model.pkl")
            model=load_object(model_path)

            with mlflow.start_run():
                

                predicted_qualities = model.predict(X_test)

                (accuracy, precision, f1, recall) = self.eval_metrics(y_test, predicted_qualities)
                
        
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("recall", recall)
                
                mlflow.log_param("model_name", type(model).__name__)
                mlflow.log_param("parameters", model.get_params())


        except Exception as e:
            raise e