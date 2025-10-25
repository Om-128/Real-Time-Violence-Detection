import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()

        #Create Artifacts directory if not exists
        os.makedirs(data_ingestion.ingestion_config.ARTIFACTS_DIR, exist_ok=True)

        # If processed files exist, load them directly
        if os.path.exists(data_ingestion.ingestion_config.FEATURES_PATH) and os.path.exists(data_ingestion.ingestion_config.LABELS_PATH):
            features = np.load(data_ingestion.ingestion_config.FEATURES_PATH)
            labels =  np.load(data_ingestion.ingestion_config.LABELS_PATH)
            print("Loaded processed features and labels from disk.")
        else:
            video_paths, labels = data_ingestion.create_dataset()
            data_transformation = DataTransformation()
            features, video_labels = data_transformation.frame_extraction(video_paths, labels)

            # Save processed features and labels to disk
            np.save(data_ingestion.ingestion_config.FEATURES_PATH, features)
            np.save(data_ingestion.ingestion_config.LABELS_PATH, video_labels)

        print(f"Extracted Features Shape: {features.shape}")
        print(f"Extracted Labels Shape: {video_labels.shape}")

    except Exception as e:
        raise CustomException(e, sys)