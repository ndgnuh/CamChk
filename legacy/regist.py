
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcStreamerState
import av
import cv2
import streamlit as st
import sys
from modules.detector import FaceDetector
from datetime import datetime
import os

DATA_PATH = '/home/dark_hold/Data/Data_sycamo'
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

with open('mattermostname.txt', 'r', encoding='utf8') as f:
    test = [s[:-1] for s in f.readlines()]
    text_input = st.selectbox("StaffID", test)
# text_input = st.text_input(
#         "Enter your name 👇",
#     )

col1, col2 = st.columns(2)
# if text_input:
#     st.write("Your name: ", text_input)

if not os.path.isdir(DATA_PATH+'/'+text_input) and (text_input != '' or text_input!="Enter your name 👇"):
            os.makedirs(DATA_PATH+'/'+text_input)

import numpy as np

from matplotlib import pyplot as plt
import threading

lock = threading.Lock()

if 'img' not in st.session_state:
    st.session_state['img'] = None
if 'reg' not in st.session_state:
    st.session_state['reg'] = False

def video_frame_callback(frame):
    img = frame.to_ndarray(format="rgb24")
    with lock:
        img = Image.fromarray(img)
        img.save("cur.png")

with col1:
    ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
    register = st.button("Register")
from PIL import Image
import requests
import json
import numpy as np
with col2:
    if register:
        with lock:
            now = datetime.now()
            now = now.strftime("%d-%m-%Y_%H-%M-%S")
            img = Image.open("cur.png")
            data = {
                "text_input": text_input,
                "user": json.dumps({ "img":np.array(img).tolist()})
            }
            r =requests.post(url='http://10.10.10.112:8000/register-user/', json = data)
            
            re = r.json()['result']
            fl = json.loads(re)
            for face in np.array(fl["face_list"]):
                im =  Image.fromarray(np.uint8(face[:,:,::-1])).convert('RGB')
                st.image(im)
