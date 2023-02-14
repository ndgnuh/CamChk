import re

import torch
import torch.nn as nn
import ast
import numpy as np
from pony.orm import db_session
from ..recognizer.models import  Embedder
from ..database import db
from functools import lru_cache

class recognize:

    def __init__(self,config):
        self.config = config
        self.emd = Embedder(self.config['embedder'])

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.emve  = db.get_all_vectors()
        self.embs_base = [np.array(ast.literal_eval(x[0])) for x in self.emve]
        self.ids_base =[int(re.findall(r'\d+',str(x[1]))[0]) for x in self.emve]

    def update_emve(self):
        self.emve  = db.get_all_vectors()
        self.embs_base = [np.array(ast.literal_eval(x[0])) for x in self.emve]
        self.ids_base =[int(re.findall(r'\d+',str(x[1]))[0]) for x in self.emve]

    def compare(self, img1, img2):
        embed_vec1 = self.emd(np.expand_dims(img1, axis=0))
        embed_vec2 = self.emd(np.expand_dims(img2, axis=0))
        score = self.cos(embed_vec1,embed_vec2)
    def __call__(self,img: np.ndarray):
        # emve = db.get_all_vectors()
        # Thread(target=self.update_emve).start()
        
        embs =  self.emd(np.expand_dims(img, axis=0))

        @lru_cache
        def is_authenticated( score):
            return score > self.config['scoreclassifier']['embedding_threshold']

        l = []
        for tmp_vec in self.embs_base:
            score =  self.cos(embs,torch.from_numpy(tmp_vec)).numpy()[0]
            l.append(score) 
        
        idx_max = np.argmax(l) 
        if is_authenticated( l[idx_max]):
            with db_session:
                staff =db.get_staff_byID(self.ids_base[idx_max])
                return staff[0][1].split()[0] , l[idx_max]
        else:
            return 'unknown', l[idx_max]



