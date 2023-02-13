import json
import time
from datetime import datetime
from threading import Thread
import threading
from collections import deque
import logging
import cv2
import base64
import numpy as np
from datetime import datetime
from icecream import ic
# from modules.process.rabbitmq import PikaProducer
# from modules.process.directions import calculate_direction, meaningful_direction, parse_direction
# from modules.camera.model.manager import Manager as ModelManager
from modules.controller.recognizerController import recognize
from ..helpers.utils import match_iou, serialize_image, denumpy_bbox
from ..detector import FaceDetector
from ..database.models import store_record
from ..database import db

detector_ = FaceDetector()


def cameraMain(name, cfg, feed_queue):
    import threading
    from argparse import Namespace
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FPS, 5)
    state = Namespace()
    state.frame = None
    state.new_frame = False

    def capture_continously():
        while camera.isOpened():
            _, frame = camera.read()
            if frame is not None:
                
                state.frame = cv2.resize(frame, (640, 480))
                # cv2.imshow('cut', state.frame)
                cv2.imwrite('save.png', state.frame)
                state.new_frame = True
        
    
    t = threading.Thread(target=capture_continously)
    t.start()
    old_frame=0 
    
    while True:
        time.sleep(0.005)
        if state.new_frame:
            timestamp = datetime.now().timestamp()
            # print(abs(state.frame-old_frame).mean()) 
            if  abs(state.frame-old_frame).mean() > 0:
                # print(abs(state.frame-old_frame).mean()) 
                 
      
                feed_queue.put((name, state.frame, timestamp))
            old_frame=state.frame

    
        # print(timestamp)
               

def  detectionMain(config, feed_queue):
    
    import random
    from queue import Queue
    rec = recognize(config)
    # queue = Queue()
    
    face_list = []
    
    def remove_face():
        threading.Timer(60.0, remove_face).start()
        if len(face_list) != 0:
            face_list.pop(0)

    Thread(target=remove_face).start()
    while True:
        
        cam_name, frame, timestamp = feed_queue.get()
        # print(f"Queue size: {feed_queue.qsize()}")
        boxes, scores, landmarks = detector_(frame)
        
        # print(boxes, scores, landmarks)
        # print(timestamp)
        try:
            cv2.imwrite(f'image.png', frame)
            # face_list = [] 
            # infor_face_list = []
            for i in range(len(scores)):
                box, score, landmark = boxes[i], scores[i], landmarks[i]
                
                now = datetime.now()
                now = now.strftime("%d-%m-%Y_%H-%M-%S")
                
                resize_image = cv2.resize(frame[box[1]: box[3],box[0]:box[2] ],(112,112))
                face = resize_image # face

                
                staff, sc = rec(face)
                    
                if staff in face_list and staff != 'unknown':
                    pass
                else:
                    print('++'*50)
                    face_list.append(staff)
                print(staff,sc)
                cv2.imwrite(f'images/{staff}_{now}.png', face)
            
                # ic()
                # print(staff,sc)
            #     # infor_face
            #     infor_face = {}
            #     infor_face['confidence'] = score
            #     infor_face['bbox'] = str(box)
            #     infor_face['landmarks'] = landmark
                
            #     infor_face['id'] = random.randint(0,100)
                
            #     # encode_img= str(base64.b64encode(resize_image))
            #     # db.store_record(encode_img,timestamp)
            #     # print(str(encode_img))
            #     # print(type(encode_img))
            #     # print(infor_face['bbox'], type(infor_face['bbox']) )
            #     # test = np.asarray(infor_face['bbox'])
            #     print(landmark, type(landmark))
            #     face_list.append(face)
            #     infor_face_list.append(infor_face)
                # cv2.imwrite(f'images/image{i}_{now}.png', resize)


            # queue.put((cam_name, frame, face_list, infor_face_list, timestamp))
        except:
            pass
    # pass