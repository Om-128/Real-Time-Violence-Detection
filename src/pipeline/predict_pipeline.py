import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException

import cv2
from tensorflow.keras.models import load_model

class PredictPipeline:
    '''
        Loads a trained model and makes predictions on new data.
    '''
    def __init__(self):
        self.model = load_model(os.path.join("artifacts", "trained_model.keras"))
        self.sequence_length = 10
        self.image_height = 224
        self.image_width = 224


    def process_frame(self, frame):
        '''Resize and normalize frame like in training.'''
        frame = cv2.resize(frame, (self.image_width, self.image_height))
        frame = (frame / 255.0).astype(np.float16)
        return frame

    def predict(self, frame_sequence):
        '''
            Predicts the class of the given frame sequence. 
        '''
        try:
            if(len(frame_sequence) != self.sequence_length):
                raise ValueError(f"Expected {self.sequence_length} frames, got {len(frames_sequence)}")
            
            ''' Prepare input for model '''
            input_data = np.expand_dims(np.array(frame_sequence, dtype=np.float16), axis=0)

            ''' Make prediction '''
            pred = self.model.predict(input_data, verbose=0)[0][0]
            label = "Violence" if pred > 0.5 else "Non-Violence"

            confidence = float(pred if pred > 0.5 else 1 - pred)
        
            return label, confidence

        except Exception as e:
            raise CustomException(e, sys)
