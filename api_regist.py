from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
import os
from random import randint
import uuid
import io
import numpy as np
from datetime import datetime
import cv2
from PIL import Image

from modules.detector import FaceDetector
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/register/")
async def regist_user(file: UploadFile = File(...)):
    contents = await file.read()  # <-- Important!
    print(type(contents))
    image = Image.open(io.BytesIO(contents))
    image.save('t.png')
    return 0
import json
import cv2
from PIL import Image

@app.post("/register-user/")
async def regist_user(user: Request):
    
    DATA_PATH = '/home/dark_hold/Data/Data_sycamo'
    contents = await user.json()  # <-- Important!
    img = json.loads(contents["user"])
    user_img = np.array(img['img'])[:,:,::-1]
    cv2.imwrite(f'test.png',  user_img)
    img = Image.open("test.png")
    detector_ = FaceDetector(keep_top_k = 1)
    boxes, scores, landmarks = detector_(np.array(img))
    face_l = []
    for i in range(len(scores)):
        # print (i)
        box, score, landmark = boxes[i], scores[i], landmarks[i]
        
        now = datetime.now()
        now = now.strftime("%d-%m-%Y_%H-%M-%S")
        resize_image = cv2.resize(np.array(img)[box[1]: box[3],box[0]:box[2] ],(112,112))
        face = resize_image[:,:,::-1] # face
        face_l.append(face)
        cv2.imwrite(f'{DATA_PATH}/{contents["text_input"]}/image{i}_{now}.png', face)
        
    return {'result':json.dumps({ "face_list":np.array(face_l).tolist()})}
    # return 0