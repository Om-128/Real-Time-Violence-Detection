import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Bidirectional, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

@dataclass
class ModelTrainerConfig:
    MODEL_SAVE_PATH = os.path.join("artifacts", "trained_model.keras")

class ModelTrainer:
    '''
        Trains a simple model on the extracted features and saves the trained model.
    '''

    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def train_model(self, SEQUENCE_LENGTH, IMAGE_HIGHT, IMAGE_WIDTH,
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

            ''' Input Layer '''
            model.add(Input(shape=(SEQUENCE_LENGTH, IMAGE_HIGHT, IMAGE_WIDTH, 3)))

            ''' Passing mobilenet in the TimeDistributed layer to handle the sequence '''
            model.add(TimeDistributed(mobileNet))

            model.add(Dropout(0.25))

            model.add(TimeDistributed(Flatten()))

            ''' Adding Bidirectional LSTM layers '''
            lstm_fw = LSTM(units=32)
            lstm_bw = LSTM(units=32, go_backwards=True)

            model.add(Bidirectional(lstm_fw, backward_layer=lstm_bw))

            model.add(Dropout(0.25))

            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.25))

            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.25))

            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.25))

            model.add(Dense(1, activation='sigmoid'))

            ''' Early Stopping and Model Checkpoint can be added here '''
            # Create Early Stopping Callback to monitor the accuracy
            early_stopping_callback = EarlyStopping(
                monitor = 'val_accuracy', 
                patience = 10,
                restore_best_weights = True
            )

            ''' Create ReduceLROnPlateau Callback to reduce overfitting by decreasing learning '''
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.6,
                patience=5,
                min_lr=0.00005,
                verbose=1
            )

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            history = model.fit(
                        x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=20,
                        batch_size=8,
                        verbose=1,
                        callbacks=[early_stopping_callback, reduce_lr]
                    )

                # Evaluate on test set
            test_loss, test_acc = model.evaluate(x_test, y_test)
            print(f"Test Accuracy: {test_acc:.4f}")

            # Save the model
            model.save(self.trainer_config.MODEL_SAVE_PATH)

            return model

        except Exception as e:
            raise CustomException(e, sys)