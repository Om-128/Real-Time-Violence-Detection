import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException

from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

@dataclass
class ModelTrainerConfig:
    MODEL_SAVE_PATH = os.path.join("artifacts", "trained_model.npy")

class ModelTrainer:
    '''
        Trains a simple model on the extracted features and saves the trained model.
    '''

    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def train_model(self, 
        x_train, x_val, x_test,
        y_train, y_val, y_test
    ):
        try:
            mobileNet = MobileNetV2(include_top=False, weights='imagenet')

            ''' Fine-Tuning to make the last 40 layer trainable '''
            mobileNet.trainable = True

            for layers in mobileNet.layers[:-40]:
                layers.trainable = False

            ''' Build Model Architecture '''
            model = Sequential()

        except Exception as e:
            raise CustomException(e, sys)