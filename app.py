from flask import Flask, render_template, Response
import cv2
from io import BytesIO
from datetime import datetime
from functools import lru_cache
import numpy as np
import copy
from modules.database import db
from modules.helpers.utils import read_config
from icecream import ic
import base64
from PIL import Image
app = Flask(__name__)



@lru_cache
def load_detector():
    from modules.detector import FaceDetector
    return FaceDetector(keep_top_k=600)


@lru_cache
def get_idx(framelist):
    return np.argmax([cv2.Laplacian(frame, cv2.CV_8UC3).var() for frame in framelist])

# num_frames = None

def generate_frames():
    camera = cv2.VideoCapture(-1)
    # camera = cv2.VideoCapture(
    #     "rtsp://admin:Grooo123@10.10.1.29:554/Streaming/Channels/1")
    # camera.set(cv2.CAP_PROP_FPS, 10)
    detector_ = load_detector()
    # global num_frames

    try:
        ic()
        frames = []
        max_frames = 16
        num_frames = 0
        cp_num_frames = copy.deepcopy(num_frames)
        old_frame = 0
        patience = 100
        patience_value = 0
        while True:

            # read the camera frame

            ic()
            success, frame = camera.read()
            # frame = copy.deepcopy(display_frame)
            
            # jpg_img = cv2.imencode('.jpg', frame)
            # cv2.imwrite('frame_s.jpg', jpg_img)
            if not success:
                break
            else:
                if frame is None:
                    continue

                if num_frames < 16:
                    frames.append(frame)
                    num_frames += 1
                    continue
                else:
                    # num_frames = cp_num_frames
                    tmp =2
                    # ic()
                # #     print(len(frames))
                # #     ic()
                # #     # idx = get_idx(frames) #np.argmax([cv2.Laplacian(frame, cv2.CV_8UC3).var() for frame in frames])
                # #     ic()
                # #     # frame = frames[idx]
                #     frames = []

                # if  abs(frame-old_frame).mean() > 0 or patience_value == patience:
                #     patience_value = 0
                #     # feed_queue.put((name, frame, timestamp))
                # else:
                #     patience_value += 1
                ic()
                boxes, scores, landmarks = detector_(frame)
                try:

                    for i in range(len(scores)):

                        box, score, landmark = boxes[i], scores[i], landmarks[i]

                        resize_image = cv2.resize(
                            frame[box[1]: box[3], box[0]:box[2]], (112, 112))
                        face = resize_image
                        jpg_img = cv2.imencode('.jpg', face)
                        enccode = base64.b64encode(jpg_img[1]).decode('utf-8')
                        # decode_str = base64.b64decode(enccode)
                        # de_img = Image.open(BytesIO(decode_str))
                        # de_img.save('face_test.jpg')
                        # enccode = base64.b64encode(face)
                        # print("------------------------------")
                        # print(enccode)
                        timestamp = datetime.now().timestamp()
                        # db.store_record(enccode, timestamp)
                except Exception as e: print(e)
                # print(boxes)
                ret, buffer = cv2.imencode('.jpg',frame)
                frame = buffer.tobytes()
                print('------------------------------')
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            ic()

    except:
        pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", ssl_context=(
        'cert.pem', 'key.pem'), debug=True, port=5000)
