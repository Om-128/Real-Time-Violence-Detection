import cv2
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolov8n.pt") 

cam = cv2.VideoCapture(1)

if not cam.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cam.read()
    
    if not ret:
        print("Failed to capture")

    result = model(frame)

    #Draw bounding boxes on persion

    for box in result[0].boxes:
        class_id = int(box.cls.cpu().numpy())
        label = result[0].names[class_id]

        if label == 'person':
            #Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, 'Om', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("DroidCam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()