import torch
import torch.nn as nn
import numpy as np
import onnxruntime as rt
# from rabbitmq import PikaProducer
from functools import lru_cache
from .utils.helpers import loadw


@lru_cache
class ScoringModel:
    def __init__(self,config: dict):
        self.embed = Embedder(config)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def __call__(self, img1: np.array, img2: np.array):

        return self.cos(self.embed(img1), self.embed(img2)).numpy()[0]
    
    def cos_(self, vec1, vec2):
        return self.cos(vec1, vec2).numpy()[0]

class Embedder:
    """
    Get vector embeding of face use Onnx model
    """

    def __init__(self, config: dict) -> None:
        self.__load_model__(weight_path=config['weightPath'],
                            fallback_url=config["fallbackUrl"])
    @lru_cache
    def __load_model__(self, weight_path: str, fallback_url: str):
        providers = [
            'TensorrtExecutionProvider',
                     'CUDAExecutionProvider',
                     'CPUExecutionProvider']
        self.onnx_model = loadw(rt.InferenceSession,
                                weight_path=weight_path,
                                fallback_url=fallback_url,
                                providers=providers)
        self.input_name = self.onnx_model.get_inputs()[0].name
        self.output_name = self.onnx_model.get_outputs()[0].name

    def preprocess(self, image):
        """
        Tranpose dimension from (batch, width, height, channel) to (batch, channel, width, height)
        """
        image = np.transpose(image, (0, 3, 1, 2))
        return (image-127.5) / 128

    def forward(self, image):
        image = self.preprocess(image).astype(np.float32)
        result = self.onnx_model.run(None, {self.input_name: image})
        return torch.from_numpy(result[0]).float()

    def __call__(self, image):
        return self.forward(image)

    def tocpu(self):
        self.onnx_model.set_providers(["CPUExecutionProvider"])
