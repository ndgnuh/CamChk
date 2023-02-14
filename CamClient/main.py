import os
import yaml
import cv2
from retinaface_detector import FaceDetector
from threading import Thread
from queue import Queue


def process_faces(boxes, scores, landmarks):
    pass


def main():
    with open("config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    detect_face = FaceDetector()
    camera = cv2.VideoCapture(config['camera'])
    while camera.isOpened():
        _, frame = camera.read()
        if frame is None:
            continue
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        boxes, scores, landmarks = detect_face(frame)
        if len(boxes) > 0:
            print(boxes, scores, landmarks)


if __name__ == "__main__":
    main()
