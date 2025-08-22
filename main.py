
from pymongo import MongoClient
import face_recognition
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import tempfile, shutil
import os
import requests
import io
import cloudinary
import cloudinary.uploader
from bson.json_util import loads
from typing import List

# ======== Cloudinary Config =========
cloudinary.config(
  cloud_name = 'dwsbud96w',
  api_key = '414542429676267',
  api_secret = 'ISn9g53tW_LNuFW2i9h-Bxkbx24'
)

# ===================== DB CONNECTION ===================== #
MONGO_URI = "mongodb+srv://7843949343akv:1gyMZedN5JedmlFh@cluster0.lzipulc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["test"]
collection = db["images"]   # <-- your image collection

# ===================== FASTAPI ===================== #
app = FastAPI()


@app.get("/")
def home():
    return JSONResponse(content={"message": "Hello from FastAPI on Vercel"})

@app.post("/match-face")
async def match_face(file: UploadFile = File(...)):
    tmp_dir  = tempfile.mkdtemp()
    tmp_path = f"{tmp_dir}/{file.filename}"
    with open(tmp_path,'wb') as f:
        shutil.copyfileobj(file.file,f)

    uploaded_img  = face_recognition.load_image_file(tmp_path)
    uploaded_encs = face_recognition.face_encodings(uploaded_img)

    if not uploaded_encs:
        shutil.rmtree(tmp_dir)
        return JSONResponse({"error":"No face detected"}, status_code=400)

    matches = []
    tolerance = 0.45

    # iterate all saved images in DB
    for doc in collection.find({}):
        # <-- Skip documents that do NOT have an encoding
        if "encoding" not in doc:
            continue

        known_enc = np.array(doc["encoding"])

        for uenc in uploaded_encs:
            dist = face_recognition.face_distance([known_enc], uenc)[0]
            if dist <= tolerance:
                matches.append({
                    "userId"  : doc["userId"],
                    "imageUrl": doc["imageUrl"],
                    "simScore": round((1 - dist) * 100, 2)
                })

    shutil.rmtree(tmp_dir)
    return {"matches": matches}


@app.get("/health")
def health():
    return {"status": "ok"}

# ===================== FASTAPI Uploade image===================== #
@app.post("/upload-multiple")
async def upload_multiple_faces(userId: str, files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        # 1 - upload to Cloudinary
        upload_result = cloudinary.uploader.upload(await file.read())
        url = upload_result["secure_url"]

        # 2 - generate face encoding
        img   = face_recognition.load_image_file(io.BytesIO(requests.get(url).content))
        encs  = face_recognition.face_encodings(img)
        if not encs:
            continue

        # 3 - save in DB
        doc = {
            "userId"   : userId,
            "imageUrl" : url,
            "encoding" : encs[0].tolist()
        }
        # collection.insert_one(doc)
        res = collection.insert_one(doc)
        doc["_id"] = str(res.inserted_id) 
        results.append(doc)

    return {"saved": results}



if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)

