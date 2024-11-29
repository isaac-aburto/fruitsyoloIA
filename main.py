from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()

    results = model.predict(frame, imgsz=640, conf=0.90)

    annotations = results[0].plot()

    cv2.imshow("DETECTION de FRUTAS", annotations)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
