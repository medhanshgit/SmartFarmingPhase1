import cv2
import torch
import numpy as np
from time import time

class RealTimeObjectDetection:
    def __init__(self, out_file):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.out_file = out_file

    def score_frame(self, frame):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        
        results = self.model(frame)
        pred = results.pred[0]  # Get the predictions
        labels = pred[:, -1].cpu().numpy()  # Extract labels
        cord = pred[:, :4].cpu().numpy()  # Extract coordinates
        return labels, cord

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] < 0.2:
                continue
            x1 = int(row[0] * x_shape)
            y1 = int(row[1] * y_shape)
            x2 = int(row[2] * x_shape)
            y2 = int(row[3] * y_shape)
            bgr = (0, 255, 0)
            classes = self.model.names
            label_font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, classes[int(labels[i])], (x1, y1), label_font, 0.9, bgr, 2)
        return frame

    def __call__(self):
        player = cv2.VideoCapture(0)
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
        ret, frame = player.read()
        while ret:
            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 3)
            print(f"Frames Per Second : {fps}")
            out.write(frame)
            ret, frame = player.read()
        player.release()
        out.release()

if __name__ == "__main__":
    out_file = 'output.avi'
    detector = RealTimeObjectDetection(out_file)
    detector()
