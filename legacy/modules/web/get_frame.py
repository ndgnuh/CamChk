import cv2
import logging
from datetime import datetime
import numpy as np
def cam_frame_buffer(l,config, feed_queue):
    from argparse import Namespace
    l.acquire()
    
    try:
        camera = cv2.VideoCapture(config['url'])
        state = Namespace()
        state.frame = None
        state.new_frame = False
        frames = []
        max_frames = 10
        num_frames = 0

    
        old_frame=0
        patience = 100
        patience_value = 0
        while True:
            success, frame = camera.read()
            if frame is None:
                continue

            if num_frames < max_frames:
                frames.append(frame)
                num_frames += 1
                continue
            else:
                num_frames = 0
                idx = np.argmax([cv2.Laplacian(frame, cv2.CV_8UC3).var() for frame in frames])

                frame = frames[idx]
                frames = []

            frame = cv2.resize(frame, (640, 480))
            timestamp = datetime.now().timestamp()

            if  abs(frame-old_frame).mean() > 0 or patience_value == patience:
                patience_value = 0
                # print(success)
                feed_queue.put((success, frame, timestamp))
            else:
                patience_value += 1
        # camera.set(cv2.CAP_PROP_FPS, 10)
        # logging.info(f"starting camera {name}")
    except Exception:
        logging.warning(f"Failed to open camera at {config['url']}")
        return
    finally:
        l.release()
    
