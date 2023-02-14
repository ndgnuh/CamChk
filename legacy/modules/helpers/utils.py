from typing import Dict
import bz2
from io import BytesIO
import json
import os
from os.path import join as joinpath
import yaml
import numpy as np
from itertools import product
import hashlib


def hash_embedd(vec:str):

    return hashlib.sha256(vec.encode('utf-8')).hexdigest()


def read_config(config_path: str):
    """
    Read config from `config_path`, the config can be in many format

    SERVER_HOST and SERVER_PORT environment variable will overwrite the
    corresponding key inside the config

    Parameter:
    -----
    config_path: str
        Path to config file
    """
    if config_path.endswith("json"):
        config = json.load(open(config_path, 'r'))
    elif config_path.endswith("yml"):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception(f"Invalid config {config_path}")

    # Config server path
    if os.getenv("SERVER_HOST"):
        config['server_host'] = os.getenv("SERVER_HOST")
    if os.getenv("SERVER_PORT"):
        config['server_port'] = os.getenv("SERVER_PORT")

    # Config data storage path
    storage_path = config['api']['data_storage']['image_directory']
    storage_path = joinpath(*storage_path.split("/"))
    config['api']['data_storage']['image_directory'] = storage_path

    return config


def load_config(path: str) -> Dict:
    if path.endswith("yml"):
        with open(path, "r") as f:
            config = yaml.load(stream=f, Loader=yaml.FullLoader)
    elif path.endswith("json"):
        with open(path, "r") as f:
            config = json.load(stream=f)
    elif path.endswith("yaml"):
        with open(path, "r") as f:
            config = yaml.load(stream=f, Loader=yaml.FullLoader)
    else:
        raise "Invalid file format"
    return config


def match_iou(frame, info_face, info_person):
    if len(info_person) == 0:
        return info_person

    def get_bbox(infos):
        bb = []
        for i in range(len(infos)):
            bb += [infos[i]['bbox']]
        return np.array(bb)

    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        # wB=boxB[2]-boxB[0]
        # boxB[0]=boxB[0]+wB/3
        # boxB[2]=boxB[2]-wB/3

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea)

        if iou > 0.4:
            p1 = np.array([boxB[0], boxB[1]])
            p2 = np.array([boxB[2], boxB[1]])
            p3 = np.array([boxA[0], boxA[1]])
            d = (np.cross(p2-p1, p3-p1)/np.linalg.norm(p2-p1))
            iou = iou+1/d+boxAArea/boxBArea

        # return the intersection over union value
        return iou

    l1 = get_bbox(info_face)
    l2 = get_bbox(info_person)

    # early return, we don't care
    # if there is no one

    ious = np.array([bb_intersection_over_union(box1, box2) for box1, box2 in product(
        get_bbox(info_face), get_bbox(info_person))]).reshape(len(l1), len(l2))

    def match_person_to_face(ious):
        # copy incase we need to use ious again
        # not necessary for now
        ious = np.copy(ious)
        m, n = ious.shape

        # Map 1-1 person to the face box
        # Column = person
        # Row = face
        mapping = {}
        for _ in range(min(m, n)):
            best_columns = np.argmax(ious, axis=1)
            confi = [ious[i, c] for (i, c) in enumerate(best_columns)]
            face_box = np.argmax(confi)
            person_box = best_columns[face_box]
            mapping[face_box] = person_box
            # ignore previously mapped columns/rows
            ious[face_box, :] = 0
            ious[:, person_box] = 0
        return mapping

    boxes = []
    mapping = match_person_to_face(ious)
    for (face_box_, person_box_) in mapping.items():
        try:
            print(f"#person: {len(info_person)}, #face {len(info_face)}")
            boxes.append(info_face[face_box_])
            # info_person[person_box_]['bbox'] = info_face[face_box_]['bbox']
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"mapping {mapping}")
            print(f"shape: {ious.shape}")
    boxes.extend([box for (i, box) in enumerate(
        info_person) if i not in mapping.values()])
    boxes.extend([box for (i, box) in enumerate(
        info_face) if i not in mapping.keys()])

    return boxes


def serialize_image(img):
    """
    Receive an image, serialize and compress it for
    JSON serialization.

    Parameter:
    -----
    img: numpy.ndarray
        Image to be serialize
    """
    b = BytesIO()
    np.save(b, img, allow_pickle=True)
    s = b.getvalue()
    s = bz2.compress(s)
    s = s.decode('latin-1')
    return s

def denumpy_bbox(bbox):
    """Convert bbox from numpy format to normal list
    Parameters
    ----------
    bbox: numpy.ndarray
        The bbox
    """
    return list([int(round(max(pos, 0))) for pos in bbox])