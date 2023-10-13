import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()
    
    into_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    L_limit = np.array([0, 0, 150])  # Setting the lower limit for white
    U_limit = np.array([255, 30, 255])  # Setting the upper limit for white
    
    w_mask = cv2.inRange(into_hsv, L_limit, U_limit)
    
    detector = cv2.bitwise_and(frame, frame, mask=w_mask)
    
    cv2.imshow('Original', frame)
    cv2.imshow('Detector', detector)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
