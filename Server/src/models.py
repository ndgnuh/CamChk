from io import BytesIO
from base64 import b64encode, b64decode
import numpy as np
import torch
from onnxruntime import InferenceSession
from gdown import cached_download
from PIL import Image
from typing import List


class FaceEmbedModel:
    """
    Get vector embeding of face use Onnx model
    """

    def __init__(
        self,
        weight: str,
        providers=('TensorrtExecutionProvider',
                   'CUDAExecutionProvider',
                   'CPUExecutionProvider')
    ) -> None:
        if weight.startswith("http"):
            weight = cached_download(weight)
        self.model = InferenceSession(weight, providers=providers)
        self.input_name = self.model.get_inputs()[0].name

    def preprocess(self, image):
        if Image.isImageType(image):
            image = np.array(image.convert("RGB"))
            # w h c -> b w h c
            image = image[None, ...]
        # b w h c -> b c w h
        image = np.transpose(image, (0, 3, 1, 2))
        return (image - 127.5) / 128

    def forward(self, image):
        image = self.preprocess(image).astype(np.float32)
        (result,) = self.model.run(None, {self.input_name: image})
        # TODO: https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
        return result[0]

    def __call__(self, image):
        return self.forward(image)

    def tocpu(self):
        self.model.set_providers(["CPUExecutionProvider"])


class FaceMatchingModel:
    def __init__(
        self,
        face_embed_model,
        embedding_vectors: np.ndarray,
        staff_ids: List[int]
    ):
        self.face_embed_model = face_embed_model
        self.emedding_vectors = embedding_vectors
        self.staff_ids = staff_ids

    def __call__(
        self,
        image
    ):
        embed = self.face_embed_model(image)


def numpy_to_bytes(x):
    io = BytesIO()
    np.save(io, x)
    bs = b64encode(io.getvalue())
    return bs


def numpy_from_bytes(bs):
    bs = b64decode(bs)
    io = BytesIO(bs)
    io.seek(0)
    return np.load(io)
