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
        
        for contour in contours:
            # Calculate the approximate polygon that fits the contour
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            
            # Get the number of vertices of the polygon
            num_vertices = len(approx)
            
            # Get the centroid of the contour
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Label the shape based on the number of vertices
            if num_vertices == 3:
                shape_label = "Triangle"
            else:
                shape_label = "Circle"
            
            # Display the shape label outside the contour
            cv2.putText(white_detected, shape_label, (cX + 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imshow('Original', frame)
        cv2.imshow('White Detector', white_detected)
        
        if cv2.waitKey(1) == 27:  # Press 'Esc' key to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
