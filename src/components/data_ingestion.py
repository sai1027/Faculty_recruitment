import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationConfig

# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    faculty_data_path: str=os.path.join('artifacts',"faculty_raw.xlsx")
    address_data_path: str=os.path.join('artifacts',"address_raw.xlsx")
   
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df=pd.read_excel('notebook/data/faculty.xlsx')
            logging.info('Read the faculty dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.faculty_data_path),exist_ok=True)
            df.to_excel(self.ingestion_config.faculty_data_path)

            df2=pd.read_excel('notebook/data/address.xlsx')
            logging.info('Read the Address dataset as dataframe')
            df2.to_excel(self.ingestion_config.address_data_path)
           
            logging.info("Data Ingestion is completed")

            return(
                self.ingestion_config.faculty_data_path,
                self.ingestion_config.address_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
    
    # train_data,test_data=obj.initiate_data_ingestion()

    # data_transformation=DataTransformation()
    # train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    # modeltrainer=ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



