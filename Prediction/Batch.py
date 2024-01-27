from src.constants import *
from src.config.configuration import *
import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import pickle
from src.utils import load_model
from sklearn.pipeline import Pipeline



PREDICTION_FOLDER="prediction_folder"
PREDICTION_CSV="prediction_csv"
PREDICTION_FILE='output.csv'
FEATURE_ENGG_FOLDER='feature_engg_folder'


ROOT_DIR=os.getcwd()
BATCH_PREDICTION=os.path.join(ROOT_DIR,PREDICTION_FOLDER,PREDICTION_CSV)
FEATURE_ENGG=os.path.join(ROOT_DIR,PREDICTION_FOLDER,FEATURE_ENGG_FOLDER)


class batch_prediction:
    def __init__(self,input_file_path, 
                 model_file_path, 
                 transformer_file_path, 
                 feature_engineering_file_path) -> None:
        
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feature_engineering_file_path = feature_engineering_file_path

    def start_batch_prediction(self):
        try:
            with open(self.feature_engineering_file_path,'rb') as f:
                feature_pipeline=pickle.load(f)

            with open(self.transformer_file_path,'rb') as f:
                processor=pickle.load(f)

                model=load_model(file_path=self.model_file_path)

            feature_engg_pipeline=Pipeline([
                ('feature_engineering',feature_pipeline)
            ])


            df=pd.read_csv(self.input_file_path)
            df=feature_engg_pipeline.transform(df)
            df.to_csv('feature_engineered_data.csv')

            FEATURE_ENGINEERED_PATH=FEATURE_ENGG
            os.makedirs(FEATURE_ENGINEERED_PATH, exist_ok=True)

            file_path=os.path.join(FEATURE_ENGINEERED_PATH,'batch_feature_engineer.csv')
            df.to_csv(file_path, exist_ok=True)


            transform_data=processor.transform(df)
            file_pth=os.path.join(FEATURE_ENGINEERED_PATH,'processor.csv')

            predictions=model.predict(transform_data)

            df_prediction=pd.DataFrame(predictions,columns=['prediction'])

            BATCH_PREDICTION_PATH=BATCH_PREDICTION
            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
            csv_path=os.path.join(BATCH_PREDICTION_PATH,'output.csv')

            df_prediction.to_csv(csv_path,index=False)
            logging.info(f'Batch prediction Done')



        except Exception as e:
             CustomException(e,sys)




