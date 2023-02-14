from modules.database import db


def store_record(img, timestamp, score):
    rec = db.store_record(img, timestamp)
    print(rec)
