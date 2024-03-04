from ultralytics import YOLO
import cv2 as cv
import easyocr
from sort import *
from util import *
reader = easyocr.Reader(['en'], gpu=False)
mot_tracker=Sort()

def read_plate(plate):
    texts=reader.readtext(plate)
    tt=''
    for t in texts:
        bbox,text,score=t
        text=text.upper().replace(' ','')
        tt+=text
    return tt


def getCar(vehicle_ids,num_plate):
    x1,y1,x2,y2,score,class_id=num_plate
    l=0
    for i in vehicle_ids:
        x1car,y1car,x2car,y2car,score=i
        if x1>x1car and x2<x2car and y1>y1car and y2<y2car:
            return [x1car,y1car,x2car,y2car,l]
        l+=1
    return [-1,-1,-1,-1,-1]

coco_model = YOLO('yolov8n.pt')
lic_det = YOLO(r'J:\NEW_PROJECTS\CarNumberPlateDetector\license_plate_detector.pt')

cap = cv.VideoCapture(r'J:\NEW_PROJECTS\CarNumberPlateDetector\sample.mp4')
frame_num = -1
vehicles=[2,3,5,6,7]
while True:
    ret, frame = cap.read()
    if not ret:
        break
    #frame=cv.resize(frame, None, fx=0.45, fy=0.45, interpolation=cv.INTER_AREA)
    frame_num += 1

    if frame_num % 1 == 0:
        detections = coco_model(frame)[0]
        licence_plates = lic_det(frame)[0]
        detected_vehicles=[]
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2,score, obj_cls = detection
            x1,y1,x2,y2,score,obj_cls=int(x1),int(y1),int(x2),int(y2),int(score),int(obj_cls),
            
            if int(obj_cls) in vehicles:
                cv.rectangle(frame, (x1, y1), (x2, y2), (3,186,252), 5)
                #cv.putText(frame, 'car', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, 2)
                detected_vehicles.append([x1,y1,x2,y2,score])
                #print(detected_vehicles)
        track_ids=mot_tracker.update(np.asarray(detected_vehicles))
        for plate in licence_plates.boxes.data.tolist():
            x1, y1, x2, y2 = int(plate[0]), int(plate[1]), int(plate[2]), int(plate[3])
            plate_car=getCar(track_ids,plate)
            pcar_x1,pcar_y1,pcar_x2,pcar_y2,pcar_id=int(plate_car[0]),int(plate_car[1]),int(plate_car[2]),int(plate_car[2]),int(plate_car[4])
            #cv.rectangle(frame, (x1, y1 ), (x2, y2), (0, 0, 255), 5)
            plate_croped = frame[y1:y2, x1:x2, :]
            plate_croped_gray = cv.cvtColor(plate_croped, cv.COLOR_BGR2GRAY)
            _, plate_threshold = cv.threshold(plate_croped_gray, 60, 255, cv.THRESH_BINARY)
            plate_text = read_license_plate(plate_threshold)
            #cv.rectangle(frame, (pcar_x1, pcar_y1), (pcar_x2, pcar_y2), (255, 0, 0), 2)
            if plate_text!=None:
                cv.rectangle(frame,(pcar_x1,pcar_y1),(pcar_x2,pcar_y1-100),(3,186,252),-1)
                cv.putText(frame, '  '+plate_text, (pcar_x1, pcar_y1 - 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 3, (255,255,255), 3)
            #print([pcar_x1,pcar_y1,pcar_x2,pcar_y2,pcar_id])
            print("Detected license plate:", plate_text)
            #cv.imshow('croped_threshold',plate_threshold)
    frame=cv.resize(frame, None, fx=0.35, fy=0.35, interpolation=cv.INTER_AREA)
    cv.imshow('output', frame)
    
    #cv.resizeWindow('output', 800, 600)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
