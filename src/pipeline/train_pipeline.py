import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        data_transformation = DataTransformation()

        #Create Artifacts directory if not exists
        os.makedirs(data_ingestion.ingestion_config.ARTIFACTS_DIR, exist_ok=True)

        # If processed files exist, load them directly
        if os.path.exists(data_ingestion.ingestion_config.FEATURES_PATH) and os.path.exists(data_ingestion.ingestion_config.LABELS_PATH):
            features = np.load(data_ingestion.ingestion_config.FEATURES_PATH)
            labels =  np.load(data_ingestion.ingestion_config.LABELS_PATH)
            print("Loaded processed features and labels from disk.")
        else:
            video_paths, labels = data_ingestion.create_dataset()
            features, labels = data_transformation.frame_extraction(video_paths, labels)

            # Save processed features and labels to disk
            np.save(data_ingestion.ingestion_config.FEATURES_PATH, features)
            np.save(data_ingestion.ingestion_config.LABELS_PATH, labels)

        ''' Split '''
        x_train, x_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=0.2, shuffle=True, random_state=42, stratify=labels
        )

        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        model_trainer = ModelTrainer()

        model = model_trainer.train_model(
            data_transformation.transformation_config.SEQUENCE_LENGTH,
            data_transformation.transformation_config.HEIGHT,
            data_transformation.transformation_config.WIDTH,
            x_train, x_val, x_test,
            y_train, y_val, y_test
        )

        model.summary()

    except Exception as e:
        raise CustomException(e, sys)