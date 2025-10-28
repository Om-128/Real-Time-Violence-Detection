import cv2
from ultralytics import YOLO
import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline
from src.exception import CustomException
from src.components.data_transformation import DataTransformation

''' Create a new YOLO model from scratch '''
model = YOLO("yolov8n-pose.pt")

''' Initialize Predict Pipeline '''
predict_pipeline = PredictPipeline()

sequence = []
label = "No Violence"
SEQUENCE_LENGTH = 10
confidence = 0.0

''' Capture Video '''
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    annotated_frame = frame.copy()
    
    ''' Resize the frame for prediction '''
    sequence.append(predict_pipeline.process_frame(frame))

    ''' Predict violence when sequence is ready '''
    if len(sequence) == SEQUENCE_LENGTH:
        label, confidence = predict_pipeline.predict(sequence)
        sequence = []

    ''' Perform pose estimation on the frame '''
    results = model(frame, verbose=False)

    ''' Set color based on violence label '''
    color = (0, 0, 255) if label == "Violence" else (255, 0, 0)

    ''' Draw keypoints and simple lines '''
    for r in results:
        
        if r.keypoints is not None:
            annotated_frame = r.plot()

            # If violence → change bounding box color to red
            if label == "Violence" and r.boxes is not None:
                for box in r.boxes.xyxy.cpu().numpy().astype(int):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    ''' Display label '''
    cv2.putText(annotated_frame, f"{label} ({confidence*100:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('YOLOv8 Pose Estimation', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()