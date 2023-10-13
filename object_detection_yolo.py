import cv2
import numpy as np

def main():
    # Load YOLO model configuration and weights
    config_path = 'yolov3.cfg'
    weights_path = 'yolov3.weights'
    net = cv2.dnn.readNet(config_path, weights_path)

    # Load YOLO class names
    classes = []
    with open('coco.names', 'r') as f:
        classes = f.read().strip().split('\n')

    # Set input layer and output layer names
    layer_names = net.getUnconnectedOutLayersNames()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Prepare input image for detection
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Perform forward pass to get detections
        detections = net.forward(layer_names)

        # Process detections
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Adjust the confidence threshold as needed
                    center_x = int(obj[0] * frame.shape[1])
                    center_y = int(obj[1] * frame.shape[0])
                    width = int(obj[2] * frame.shape[1])
                    height = int(obj[3] * frame.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) == 27:  # Press 'Esc' key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
