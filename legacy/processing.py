import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seam_carving
from sklearn.preprocessing import normalize
from torchvision import datasets
from torchvision.transforms import transforms
from modules.helpers.utils import  hash_embedd
from modules.detector import FaceDetector
from modules.recognizer.models import Embedder
import shutil

class Processing:
    def __init__(self, config):
        self.config = config
        self.detection = FaceDetector()
        self.embedder = Embedder(self.config['embedder'])

    def faceFolder2Vec(self, DATA_PATH):
        """
        Get vecto embeding from data path
        """
        # embed = Embedder(self.config['embedder'])
        name_folder = DATA_PATH.split('/')[-1]
        detector_ = FaceDetector()
        dir_id_list =  [DATA_PATH+'/'+ x for x in os.listdir(DATA_PATH)]
        vec_l = []
        hash_l = []
        for path in dir_id_list:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            boxes, scores, landmarks = detector_(image)
            for i in range(len(scores)):
                box, score, landmark = boxes[i], scores[i], landmarks[i]
                face = cv2.resize(image[box[1]: box[3],box[0]:box[2] ],(112,112))
                vec = str(self.embedder(np.expand_dims(face, axis=0)).numpy().tolist()) # vec_embedding
                hash_code  = hash_embedd(vec)
                vec_l.append(vec)
                hash_l.append(hash_code)

        return name_folder,vec_l, hash_l
    
    def faceCutFolder2Vec(self, DATA_PATH):
        """
        Get vecto embeding from data path
        """
        # embed = Embedder(self.config['embedder'])
        name_folder = DATA_PATH.split('/')[-1]
        dir_id_list =  [DATA_PATH+'/'+ x for x in os.listdir(DATA_PATH)]
        vec_l = []
        hash_l = []
        for path in dir_id_list:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            vec = str(self.embedder(np.expand_dims(image, axis=0)).numpy().tolist()) # vec_embedding
            hash_code  = hash_embedd(vec)
            vec_l.append(vec)
            hash_l.append(hash_code)

        return name_folder,vec_l, hash_l
