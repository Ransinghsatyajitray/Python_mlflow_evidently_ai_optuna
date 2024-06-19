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
import pickle
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

        

            # mlflow.set_registry_uri("https://dagshub.com/ransingh.satyajit/endtoend.mlflow")
            
            #tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            #print(tracking_url_type_store)



            with mlflow.start_run():
                

                predicted_qualities = model.predict(X_test)

                (accuracy, precision, f1, recall) = self.eval_metrics(y_test, predicted_qualities)
                
        
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("recall", recall)
                
                mlflow.log_param("model_name", type(model).__name__)



                """
                # this condition is for the dagshub
                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                # it is for the local 
                else:
                    mlflow.sklearn.log_model(model, "model")
                """            
        except Exception as e:
            raise e