from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcStreamerState
import av
import cv2
import streamlit as st
import sys
from modules.detector import FaceDetector
from datetime import datetime
from icecream import ic
import os
import threading
from threading import Thread
from modules.controller.recognizerController import recognize


from modules.helpers.utils import read_config


# Initialization
if 'key' not in st.session_state:
    st.session_state['key'] = 'value'

if 'staff' not in st.session_state:
    st.session_state['staff'] = 'P'

config = read_config('./config/config.yml')

col1, col2 = st.columns(2)

rec = recognize(config)
lock = threading.Lock()

face_list = []
    
def remove_face():
    threading.Timer(60.0, remove_face).start()
    if len(face_list) != 0:
        face_list.pop(0)

Thread(target=remove_face).start()

def update_staff(staff):
    ic()
    st.session_state.staff = staff
    print(st.session_state.staff)
    with col2:
    # while True:
        st.write(staff)

# increment = st.button('Increment')

def video_frame_callback(frame):
    img = frame.to_ndarray(format="rgb24")
    detector_ = FaceDetector()
    boxes, scores, landmarks = detector_(img)
    try:
        # face_list = [] 
        # infor_face_list = []
        
        for i in range(len(scores)):
            
            box, score, landmark = boxes[i], scores[i], landmarks[i]
            
            
            resize_image = cv2.resize(img[box[1]: box[3],box[0]:box[2] ],(112,112))
            face = resize_image # face
            staff, sc = rec(face)
            now = datetime.now()
            now = now.strftime("%d-%m-%Y_%H-%M-%S")
            if staff in face_list and staff != 'unknown':
                pass
            else:
                # print('++'*50)
                face_list.append(staff)
                
                # if increment:
                #     with lock:
                #         update_staff(staff)
                print(staff,now, sc)

    except:
            pass
    # print(boxes)s
    # with lock:
    #     img = Image.fromarray(img)
    #     img.save("cur.png")

with col1:
    ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

with col2:
    # while True:
    
    st.write(st.session_state["staff"])