import cv2 as cv
import matplotlib.pyplot as plt


harcascade = r'model\haarcascade_russian_plate_number.xml'
cap=cv.VideoCapture(0)
cap.release()
cv.destroyAllWindows()
model_casc=cv.CascadeClassifier(harcascade)
count=0
while True:
    s,img=cap.read()


    img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    plate=model_casc.detectMultiScale(img_gray,1.1,4)
    
    for (x,y,w,h) in plate:
        area=w*h
        if area> 500:
            cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)
            cv.putText(img,"Number Plate",(x,y-5),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,255),2)
            print('this is numberplate')
            img_roi=img[y:y+h,x:x+w]
            cv.imshow('roi',img_roi)
           
            # Display the recognized text on the image
           

    cv.imshow("Res",img)

    if cv.waitKey(1) & 0xFF == ord('s'):
            # Save the detected plate image
            cv.imwrite('plates/img' + str(count) + '.jpg', img_roi)
            count += 1

    # Check for 'q' key press to quit the program
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv.destroyAllWindows()