import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    BASE_DIR = r"O:\AI_ML\Github\Real-Time-Violence-Detection\data\Real Life Violence Dataset"
    CATAGORIES = ["Violence", "NonViolence"]
    ARTIFACTS_DIR = os.path.join("artifacts")  # folder to save processed data
    FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "processed_features.npy")
    LABELS_PATH = os.path.join(ARTIFACTS_DIR, "processed_labels.npy")

class DataIngestion:
    """
        Walks through dataset folders and creates in-memory lists
        of video file paths and corresponding labels.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def create_dataset(self):
        try:
            video_paths = []
            labels = []

            '''Walk through each category directory (In this case: Violence, NonViolence)'''
            for class_names in self.ingestion_config.CATAGORIES:
                class_path = os.path.join(self.ingestion_config.BASE_DIR, class_names)

                if not os.path.exists(class_path):
                    raise CustomException(f"Directory {class_path} does not exist.")
                    continue
            
                '''Walk through each file in the class directory'''

                for file_name in os.listdir(class_path):
                    if file_name.endswith(('.mp4', '.avi', '.mov')):
                        video_paths.append(os.path.join(class_path, file_name))
                        labels.append(1 if class_names == "Violence" else 0)

            print(f"Total Videos Found: {len(video_paths)}")
            print(f"Total Voilence Labels: {labels.count(1)} | Total Non-Violence Labels: {labels.count(0)}")

            return video_paths, labels
        
        except Exception as e:
            raise CustomException(e, sys)


