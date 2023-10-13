import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # lower_white = np.array([0, 0, 150])
        # upper_white = np.array([255, 30, 255])
        # blue
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        white_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.erode(white_mask, kernel, iterations=1)
        white_mask = cv2.dilate(white_mask, kernel, iterations=1)
        
        white_detected = cv2.bitwise_and(frame, frame, mask=white_mask)
        
        # Contour Detection
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(white_detected, contours, -1, (0, 255, 0), 2)
        
        cv2.imshow('Original', frame)
        cv2.imshow('White Detector', white_detected)
        
        if cv2.waitKey(1) == 27:  # Press 'Esc' key to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
