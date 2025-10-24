import cv2
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolov8n-pose.pt")

# Capture Video
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform pose estimation on the frame
    results = model(frame)

    # Get keypoints of in each frame and plot them
    for r in results:
        if r.keypoints is not None:
            anotated_frame = r.plot()

    cv2.imshow('YOLOv8 Pose Estimation', anotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()