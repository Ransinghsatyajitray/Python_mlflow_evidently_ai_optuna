import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.irisclassification.logger import logging
from src.irisclassification.exception import customexception

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score
)

from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        # getting the artifact folder
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # opening the pkl file and dumping the object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    
    
# The model is using GridSearchCV.
def evaluate_model(models_clf, param_grids, X_train, y_train, X_test, y_test):
    
    try:
        model_list_clf = []
        results_clf = []
        
        
        for name, model in models_clf.items():
            print("--"*20 + f"Evaluating {name}"+"--"*20)
            # Grid Search multi class
            grid_search = GridSearchCV(model, param_grid=param_grids[name], scoring ='f1_weighted', cv=10) 
            grid_search.fit(X_train, y_train)

            # Best Model & Predictions
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            # Evaluation for multiclass classification
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print("Best Parameters:", grid_search.best_params_)
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)
            
            # Store results for comparison
            model_list_clf.append(name)
            results_clf.append({'Model': name, 'Best Parameters': grid_search.best_params_, 
                            'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1})
        
        # Create a DataFrame for easy comparison
        results_df = pd.DataFrame(results_clf)    
            
        # Find the best classification model
        best_clf_model_idx = results_df['F1 Score'].idxmax()
        best_clf_model = results_df.iloc[best_clf_model_idx]
        
        # Get predictions from the best model
        best_model_name = best_clf_model['Model']
        best_model = models_clf[best_model_name].set_params(**best_clf_model['Best Parameters'])
        best_model.fit(X_train, y_train)
        best_model_predictions = best_model.predict(X_test)
        
        # The best model is
        print(f"The best model is {best_clf_model}")
          
        
        return {"model_list_clf": model_list_clf, 
                "results_clf": results_clf, 
                "results_df": results_df, 
                "best_clf_model_idx": best_clf_model_idx, 
                "best_clf_model": best_clf_model,
                "best_model_predictions": best_model_predictions}
    
    except Exception as e:
        logging.info('Exception occured during model training')
        raise customexception(e,sys)


    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customexception(e,sys)