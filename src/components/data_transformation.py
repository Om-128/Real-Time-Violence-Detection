import os
import sys
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    SEQUENCE_LENGTH = 15  # Number of frames per video sequence
    HEIGHT = 224  # Frame height
    WIDTH = 224   # Frame width

class DataTransformation:
    """
        Transforms raw video data into sequences of frames suitable for model input.
    """
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def frame_extraction(self, video_paths, labels):
        try:
            features = []
            video_labels = []

            for index, video in enumerate(tqdm(video_paths, desc="Processing videos")):

                frame_list = []

                #Read the video file
                video_reader = cv2.VideoCapture(video)
                # Calculate the total video frames count.
                total_video_frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
                # Calculate the the interval after which frames will be added to the list.
                frame_interval = max(int(total_video_frame_count / self.transformation_config.SEQUENCE_LENGTH), 1)

                #Iterate through the video and extract frames
                for frame_counter in range(self.transformation_config.SEQUENCE_LENGTH):
                    #Set the video reader to the correct frame position
                    video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * frame_interval)
                    #Read the frame
                    success, frame = video_reader.read()

                    if not success:
                        break
                    
                    #Resize the frame
                    resized_frame = cv2.resize(frame, (self.transformation_config.WIDTH, self.transformation_config.HEIGHT))
                    #Normalize the frame
                    normalized_frame = resized_frame / 255.0
                    #Append the frame to the frame list
                    frame_list.append(normalized_frame)
                
                video_reader.release()

                if len(frame_list) == self.transformation_config.SEQUENCE_LENGTH:
                    features.append(frame_list)
                    video_labels.append(labels[index])

            return np.array(features), np.array(video_labels)

        except Exception as e:
            raise CustomException(e, sys)
