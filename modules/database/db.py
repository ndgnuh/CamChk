from pony import orm
from datetime import datetime
from shutil import copy
from collections import namedtuple
from pony.orm.core import Entity
import datetime as dt
import cv2
import numpy as np
# from modules.helpers.utils import  read_config
import os

database_file = "cam.sqlite"
db = orm.Database()
db.bind(provider='sqlite', filename=database_file, create_db=True)


class Record(db.Entity):
    ID = orm.PrimaryKey(int, auto=True)
    img = orm.Required(str)  # cru
    insert_time = orm.Required(datetime, default=datetime.now)
    capture_time = orm.Required(datetime)  # cru
    flag = orm.Required(bool, default=False)
    details = orm.Set("Record_detail")


class Record_detail(db.Entity):
    record = orm.Required(Record)
    staff_id = orm.Required(str, default='unknown')  # cru
    confidence = orm.Required(float, default=0.0)


class Personal_info(db.Entity):
    staff_id = orm.PrimaryKey(int, auto=True)  # cru
    name = orm.Required(str)
    embedd_vectors = orm.Set("Embedding_vector")


class Embedding_vector(db.Entity):
    hash_code = orm.Required(str)
    embed_vector = orm.Required(str)
    # personal_id = orm.Required(str)
    person = orm.Required(Personal_info)


db.generate_mapping(check_tables=True, create_tables=True)


@orm.db_session
def store_record(img, timestamp):

    r = Record(img=img, capture_time=datetime.fromtimestamp(timestamp))
    Record_detail(record =r)
    orm.commit()


@orm.db_session
def store_embed(name_folder, vec_l, hash_l):

    p = Personal_info(name=name_folder)
    for i in range(len(vec_l)):
        Embedding_vector(hash_code=hash_l[i], embed_vector=vec_l[i], person=p)
    orm.commit()


@orm.db_session
def get_all_vectors():
    recs = orm.select(
        (r.embed_vector, r.person)
        for r in Embedding_vector)

    return recs[:]


@orm.db_session
def get_all_staffs():
    recs = orm.select(
        (p.staff_id, p.name)
        for p in Personal_info)

    return recs[:]


def get_staff_byID(id: int):
    recs = orm.select(
        (p.staff_id, p.name)
        for p in Personal_info if p.staff_id == id)

    return recs[:]
