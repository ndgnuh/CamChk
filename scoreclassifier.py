import cv2
import threading
from threading import Thread
import multiprocessing as mp
from datetime import datetime
import time
from modules.helpers.utils import read_config, hash_embedd
# from processing import Processing
from functools import lru_cache
from modules.database import db


config = read_config('./config/config.yml')
vid = cv2.VideoCapture(0)


def load_detector_(l, name:str):
    l.acquire()

    face_list = []
    
    @lru_cache
    def remove_face():
        threading.Timer(60.0, remove_face).start()
        if len(face_list) != 0:
            face_list.pop(0)

    Thread(target=remove_face).start()

    try:
        prev_frame_time = 0
  
        # used to record the time at which we processed current frame
        new_frame_time = 0
        from modules.detector import FaceDetector
        from modules.controller.recognizerController import recognize
        rec = recognize(config)
        detector_ = FaceDetector(keep_top_k =600)
        while(True):
        
            ret, frame = vid.read()

            # Display the resulting frame
            cv2.imshow('frame', frame)
            # print(len(func_list))
            boxes, scores, landmarks = detector_(frame ) 
            try:

                for i in range(len(scores)):
                    
                    box, score, landmark = boxes[i], scores[i], landmarks[i]
                    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
                    
                    cv2.rectangle(frame, (x0, y0), (x1 , y1), (255,0,0), 4)
                    
                    # print(boxes)
                    resize_image = cv2.resize(frame[box[1]: box[3],box[0]:box[2] ],(112,112))
                    face = resize_image # face
                    staff, sc = rec(face)
                    now = datetime.now()
                    now = now.strftime("%d-%m-%Y_%H-%M-%S")
                    cv2.putText(frame, f'{staff}: {round(sc,2)} ' , (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
                    # print(staff,now, sc)
                    cv2.imshow('frame', frame)
                    
                    if staff in face_list and staff != 'unknown':
                        pass
                    else:
                        # print('++'*50)
                        face_list.append(staff)
                        
                        # if increment:
                        #     with lock:
                        #         update_staff(staff)
                        if staff != 'unknown':
                         print(staff,now, sc)

            except:
                    pass
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
    finally:
        l.release()
    

def main():
    lock = mp.Lock()
    procs = []
    # proc1 = mp.Process(target=load_recog, args=(lock,"load_detector"))  # instantiating without any argument
    # procs.append(proc1)
    proc = mp.Process(target=load_detector_, args=(lock,"load_recog"))  # instantiating without any argument
    procs.append(proc)
    try:
        for process in procs:
            process.start()

        for process in procs:
            process.join()
    except KeyboardInterrupt:
        pass
    finally:
        for process in procs:
            process.terminate()



if __name__ == "__main__":


    main()

