import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("videos/cars.mp4")
mask = cv2.imread("new-mask.png")


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter",]

limits = [310, 367, 673, 367]  #for the line
totalCount = []

#tracker
tracker = Sort(max_age=24, min_hits=3, iou_threshold=0.3 )

#creating the model
model = YOLO("../Yolo-Weights/yolov8l.pt")



while True:
    success , img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)

    #adding the car-counter img
    imgGraphics = cv2.imread("counter-pic.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    # feeding the model our masked vid/img
    results = model(imgRegion, stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # making the confidence into integers
            config = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass=="car" or currentClass=="truck" or currentClass=="motorbike" or currentClass=="bus"   and   config>0.3 :
                #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)

                # cvzone.putTextRect(img, f'{currentClass} {config}', (max(0, x1), max(35, y1)), thickness=1, scale=0.6,
                #                    offset=3)  # 35 in y so that we can see the config even when the obj goes a bit out

                currentArray = np.array([x1, y1, x2, y2,config])
                detections = np.vstack((detections , currentArray))

    resultTracker = tracker.update(detections) #giving the tracker updated array and storing results from it

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 4) #red line

    for result in resultTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2 , colorR = (255, 0, 0))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        # putting circles in the center of vehicles
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        #counting (inside a region close to the line)
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id)==0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5) #green line

    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("Image",img)  #used to show frames captured above
    #cv2.imshow("ImgRegion", imgRegion)
    cv2.waitKey(1)