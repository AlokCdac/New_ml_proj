from src.constants import *
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import *
import os, sys
from dataclasses import dataclass
from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from src.utils import save_obj
from src.config.configuration import PREPROCESSING_OBJ_FILE,TRANSFORM_TRAIN_FILE_PATH,TRANSFORM_TEST_FILE_PATH,FEATURE_ENGG_OBJ_FILE_PATH
#feature engg class
class Feature_engineering(BaseEstimator,TransformerMixin):
    def __init__(self):
        logging.info("*****************feature_engineering****************")

    def distance_numpy(self, df,lat1,lon1,lat2,lon2):
        p=np.pi/180
        a=0.5-np.cos((df[lat2]-df[lat1]*p)/2) + np.cos(df[lat1]*p) * np.cos(df[lat2]*p) * (1-np.cos((df[lon2]-df[lon1])*p))/2
        df["distance"]=12734 * np.arccos(np.sort(a))


    def transform_data(self, df):
        try:
            df.drop(["ID"],axis=1, inplace=True)
            self.distance_numpy(df,'Restaurant_latitude','Restaurant_longitude', 
                                'Delivery_location_latitude','Delivery_location_longitude')
            df.drop(['Delivery_person_ID','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude',
             'Order_Date','Time_Orderd','Time_Order_picked'],axis=1,inplace=True)
            logging.info("Dropping xcolumns from original dataset")
            return df
            
        except Exception as e:
            raise CustomException(e,sys) from e
    
    def fit(self,X,y=None):
        return self
    
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            transformed_df=self.transform_data(X)
                
            return transformed_df
        except Exception as e:
            raise CustomException(e,sys) from e
@dataclass       
class DataTransformationConfig():
    preprocessed_obj_file_path=PREPROCESSING_OBJ_FILE
    transform_train_path=TRANSFORM_TRAIN_FILE_PATH
    transform_test_path=TRANSFORM_TEST_FILE_PATH
    feature_engg_file_path=FEATURE_ENGG_OBJ_FILE_PATH

class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformation_obj_file(self):
        try:
            # defining the ordinal data ranking
            Road_traffic_density=['Low','Medium','High','Jam']
            Weather_conditions=['Sunny','Cloudy','Windy','Fog','Sandstorms','Stormy']

            # defining the categorical and numerical column
            categorical_column=['Type_of_order','Type_of_vehicle','Festival','City']
            ordinal_encod=['Road_traffic_density','Weather_conditions']
            numerical_column=['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition','multiple_deliveries','distance']

            # numerical pipeline

            numerical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='constant',fill_value=0)),
                ('scaler',StandardScaler(with_mean=False))
                ])

            # categorical pipeline

            categorical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder(handle_unknown='ignore')),
                ('scaler',StandardScaler(with_mean=False))
                ])




            # ordinal pipeline

            ordianl_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('ordinal',OrdinalEncoder(categories=[Road_traffic_density,Weather_conditions])),
                ('scaler',StandardScaler(with_mean=False))   
                ])

            processor=ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_column),
                ('categorical_pipeline',categorical_pipeline,categorical_column),
                ('ordinal_pipeline',ordianl_pipeline,ordinal_encod)

            ])
            return processor
            logging.info('Pipeline Steps Completed')

        except Exception as e:
            raise CustomException(e,sys)
        
    def get_feature_engg_obj(self):
            try:
                feature_engineering=Pipeline(steps=[('fe', Feature_engineering())])
                return feature_engineering
            
            except Exception as e:
                raise CustomException(e,sys) from e
            

    def initiate_data_transformation(self,train_path,test_path):
            try:
                train_df=pd.read_csv(train_path)
                test_df=pd.read_csv(test_path)


                logging.info("logging fe objects")

                fe_obj = self.get_feature_engg_obj()
                train_df = fe_obj.fit_transform(train_df)
                #train_df = fe_obj.fit_transform(train_df)
                logging.info(">>>" * 20 + " Test data " + "<<<" * 20)
                logging.info(f"Feature Enineering - Test Data ")
                test_df = fe_obj.transform(test_df)

                train_df.to_csv('train_data.csv')
                test_df.to_csv('test_data.csv')


                processinfo_obj=self.get_data_transformation_obj_file()

                target_column_name = 'Time_taken (min)'
            #drop_columns = [target_column_name,'id']

                X_train = train_df.drop(columns=target_column_name,axis=1)
                y_train=train_df[target_column_name]
                X_test=test_df.drop(columns=target_column_name,axis=1)
                y_test=test_df[target_column_name]



                X_train=processinfo_obj.fit_transform(X_train)            
                X_test=processinfo_obj.transform(X_test)
                logging.info("Applying preprocessing object on training and testing datasets.")
                logging.info(f"shape of {X_train.shape} and {X_test.shape}")
                logging.info(f"shape of {y_train.shape} and {y_test.shape}")
            

                logging.info("transformation completed")

                train_arr = np.c_[X_train, np.array(y_train)]
                test_arr = np.c_[X_test, np.array(y_test)]

                logging.info("train_arr, test_arr completed")

            

                logging.info("train arr , test arr")


                df_train=pd.DataFrame(train_arr)
                df_test=pd.DataFrame(test_arr)


                os.makedirs(os.path.dirname(self.data_transformation_config.transform_train_path), exist_ok=True)
                df_train.to_csv(self.data_transformation_config.transform_train_path, index=False, header=True)


                os.makedirs(os.path.dirname(self.data_transformation_config.transform_test_path), exist_ok=True)
                df_test.to_csv(self.data_transformation_config.transform_test_path, index=False, header=True)

                save_obj(file_path=self.data_transformation_config.preprocessed_obj_file_path,
                         obj=fe_obj)


                save_obj(file_path=self.data_transformation_config.feature_engg_file_path,
                         obj=fe_obj)
                
                return train_arr, test_arr, self.data_transformation_config.preprocessed_obj_file_path



            except Exception as e:
                raise CustomException(e,sys)

        



        






       