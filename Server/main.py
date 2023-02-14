import uvicorn
import database as db
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from src.models import (
    FaceEmbedModel,
    numpy_to_bytes,
    numpy_from_bytes
)
from PIL import Image
print(db)
production = False

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

embed_face = FaceEmbedModel("arcfaces_resnet100.onnx")


@app.post("/register/embedding")
@db.db_session
def register(staff_id: int, face: UploadFile = File(...)):
    image = Image.open(face.file)
    embed = embed_face(image)
    staff = db.orm.select(staff for staff in db.StaffInfo
                          if staff.staff_id == staff_id)
    if len(staff) == 0:
        return None

    assert len(staff) < 2
    staff = staff.get()
    staff.embed_vector = numpy_to_bytes(embed).decode("utf-8")
    db.orm.commit()
    return staff.to_dict()


@app.post("/query")
@db.db_session
def query(face: UploadFile = File(...), threshold: float = 0.5):
    import numpy as np
    from scipy.spatial.distance import cosine
    image = Image.open(face.file)
    embed = embed_face(image)
    known_staffs = db.orm.select(staff for staff in db.StaffInfo
                                 if len(staff.embed_vector) > 0)
    known_staffs = known_staffs[:]
    known_embeds = [
        numpy_from_bytes(staff.embed_vector.encode())
        for staff in known_staffs
    ]
    scores = [1 - cosine(embed, ref_embed) for ref_embed in known_embeds]
    idx = np.argmax(scores)
    score = scores[idx]
    if score < threshold:
        staff = None
    else:
        staff = known_staffs[idx].to_dict()
    return dict(score=score, staff=staff)


@ app.post("/debug/add_staff")
@ db.db_session
async def add_staff(rq: Request):
    body = await rq.json()
    staff_id = body["staff_id"]
    name = body["name"]
    new_staff = db.StaffInfo(staff_id=staff_id, name=name)
    return new_staff.to_dict()


@ app.post("/debug/get_embedding")
def debug_get_embedding(face: UploadFile = File(...)):
    image = Image.open(face.file)
    embed = embed_face(image)
    return embed.tolist()


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, port=8989)
