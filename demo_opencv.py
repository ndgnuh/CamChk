
# import the opencv library
import cv2
import multiprocessing as mp
import threading 
from threading import Thread
from modules.helpers.utils import read_config
from functools import lru_cache
from datetime import datetime
from modules.detector import FaceDetector
from modules.controller.recognizerController import recognize
    # global detector_

detector_ = FaceDetector(keep_top_k =750)
# define a video capture object
config = read_config('./config/config.yml')
vid = cv2.VideoCapture(config["camera"]["cam2"]["url"])


rec = recognize(config)
# detector_ = None
# rec = None

# @lru_cache
# def load_detector():

#     from modules.detector import FaceDetector
#     global detector_

#     detector_ = FaceDetector(keep_top_k =600)

# @lru_cache
# def load_recognizer():
#     global rec
#     from modules.controller.recognizerController import recognize
#     rec = recognize(config)
 
def main():

    face_list = []

    @lru_cache
    def remove_face():
        threading.Timer(60.0, remove_face).start()
        if len(face_list) != 0:
            face_list.pop(0)

    Thread(target=remove_face).start()

    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
    
        # Display the resulting frame
        try:

            cv2.imshow('frame', frame)
        except:
            pass
        boxes, scores, landmarks = detector_(frame ) 
        try:

            for i in range(len(scores)):
                
                box, score, landmark = boxes[i], scores[i], landmarks[i]
                x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
                

                
                
                # print(boxes)
                resize_image = cv2.resize(frame[box[1]: box[3],box[0]:box[2] ],(112,112))
                face = resize_image # face
                staff, sc = rec(face)
                now = datetime.now()
                now = now.strftime("%d-%m-%Y_%H-%M-%S")
                cv2.rectangle(frame, (x0, y0), (x1 , y1), (255,0,0), 4)
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
                        # pass
                        print(staff,now, sc)
                    else:

                        cv2.imwrite(f'images/{staff}{now} score:{ round(sc,2)}.png', face)
                
        
        except:
            pass
        # print(boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # proc = Thread(target=load_detector)
    # proc.start()
    # proc.join()
    main()
