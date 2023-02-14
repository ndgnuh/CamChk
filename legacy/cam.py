from modules.helpers.utils import read_config
from modules.controller.cameraController import cameraMain , detectionMain
import multiprocessing as mp
feed_queue: mp.Queue = mp.Queue()




def main():
    config = read_config('./config/config.yml')
    processQueue = []
    for (name, cfg) in config['camera'].items():
        process = mp.Process(target=cameraMain, args=(name, cfg,feed_queue))
        processQueue.append(process)

    detection_procs = 1
    for _ in range(detection_procs):
        process = mp.Process(target=detectionMain, args=(config, feed_queue))
        processQueue.append(process)

    print(processQueue)

    try:
        for process in processQueue:
            process.start()

        for process in processQueue:
            process.join()
    except KeyboardInterrupt:
        pass
    finally:
        for process in processQueue:
            process.terminate()

if __name__ == "__main__":
    main()