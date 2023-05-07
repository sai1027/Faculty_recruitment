import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor_model.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            logging.info("pkl loaded")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)

            logging.info("got preds ")

            main_df=pd.read_excel("artifacts/final_data.xlsx")
            main_df=main_df[main_df["cluster"]==preds[0]]

            logging.info("main df cluster filtered")
            main_df.to_excel("artifacts/output.xlsx")
            logging.info("main exlxel created..")
            
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(  self,
            GENDER:int,
            AGE:int,
            TETSCORE:int,
            LEVEL:int,
            EXPERIENCE:int,
            SUBJECT:str,
            ZONE:int
            ):

        self.GENDER = GENDER

        self.AGE = AGE

        self.TETSCORE = TETSCORE

        self.LEVEL = LEVEL

        self.EXPERIENCE = EXPERIENCE 

        self.SUBJECT = SUBJECT

        self.ZONE = ZONE

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                
                "AGE": [self.AGE],
                "TETSCORE": [self.TETSCORE],
                "EXPERIENCE": [self.EXPERIENCE],
                "ZONE_"+self.ZONE:1,
                "LEVEL": [self.LEVEL],
                "GENDER": [self.GENDER],
                "SUBJECT_"+self.SUBJECT: 1,
                
            }

            def fun(custom_data_input_dict):
                di={'AGE':0, 'TETSCORE':0, 'EXPERIENCE':0,"LEVEL":0,'GENDER':0,'ZONE_1':0, 'ZONE_2':0, 'ZONE_3':0, 'ZONE_4':0,'ZONE_5':0,
                    'SUBJECT_ENGLISH':0,'SUBJECT_MATHS':0,'SUBJECT_SCIENCE':0,'SUBJECT_SOCIAL':0,'SUBJECT_TELUGU':0}
                for i in custom_data_input_dict:
                    di[i]=custom_data_input_dict[i]
                
                
                return pd.DataFrame(di)

            df3=fun(custom_data_input_dict)
            logging.info("predict dataframe created")
            return df3

            

        except Exception as e:
            raise CustomException(e, sys)

