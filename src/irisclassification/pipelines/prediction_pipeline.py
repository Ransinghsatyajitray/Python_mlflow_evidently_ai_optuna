import os
import sys
import pandas as pd
from src.irisclassification.exception import customexception
from src.irisclassification.logger import logging
from src.irisclassification.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            preprocessor_path = os.path.join(".", "artifacts", "preprocessor.pkl")
            model_path = os.path.join(".", "artifacts", "model.pkl")
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            scaled_data = preprocessor.fit_transform(features)
            
            pred = model.predict(scaled_data)
            
            return pred[0]                                   
        
        except Exception as e:
            raise customexception(e, sys)
    
       
class CustomData:
    def __init__(self,
                 SepalLengthCm:float,
                 SepalWidthCm:float,
                 PetalLengthCm:float,
                 PetalWidthCm:float):
        
        self.SepalLengthCm = SepalLengthCm
        self.SepalWidthCm = SepalWidthCm
        self.PetalLengthCm = PetalLengthCm
        self.PetalWidthCm = PetalWidthCm
            
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'SepalLengthCm':[self.SepalLengthCm],
                    'SepalWidthCm':[self.SepalWidthCm],
                    'PetalLengthCm':[self.PetalLengthCm],
                    'PetalWidthCm':[self.PetalWidthCm]
                }
                df = pd.DataFrame([custom_data_input_dict])
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception occured in prediction pipeline')
                raise customexception(e,sys)