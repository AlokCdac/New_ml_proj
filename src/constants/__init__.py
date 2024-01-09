import os , sys
from datetime import datetime


def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d %H,%m,%S')}"


CURRENT_TIME_STAMP=get_current_time_stamp()

# Root Directory 
ROOT_DIR_KEY = os.getcwd()

# Data file path 
DATA_DIR = "Data"
DATA_DIR_KEY = 'finalTrain.csv'

ARTIFACT_DIR_KEY='Artifact'

DATA_INGESTION_KEY = 'data_ingestion'
DATA_INGESTION_RAW_DATA_DIR_KEY= 'raw_data_dir'
DATA_INGESTION_INGESTED_DIR_NAME_KEY= 'ingested_dir'
RAW_DATA_DIR_KEY = 'raw.csv'
TRAIN_DATA_DIR_KEY = 'train.csv'
TEST_DATA_DIR_KEY = 'test.csv' 


#data transformation related variable 
DATA_TRANSFORMATION_ARTIFACT='data_transformation'
DATA_PREPROCESSED_DIR="PROCESSOR"
DATA_TRANSFORMATION_PROCESSING_OBJ="processor.pkl"
DATA_TRANSFORM_DIR="TRANSFORMATION"
TRANSFORM_TRAIN_DIR_KEY="train.csv"
TRANSFORM_TEST_DIR_KEY="test.csv"



#Model Training 
MODEL_TRAINER_KEY="model_trainer"
MODEL_OBJECT='model.pkl'