from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
import numpy as np
import cv2
import os

app = FastAPI()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
dataset_path = "./face_dataset/"
MIN_SAMPLES = 20

########## KNN CODE ############
def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])

    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]
################################


############# DATASET LOADER ################
def load_dataset():
    face_data = []
    labels = []
    class_id = 0
    names = {}

    for fx in os.listdir(dataset_path):
        if fx.endswith('.npy'):
            employee_id = fx[:-4]
            names[class_id] = employee_id

            data_item = np.load(os.path.join(dataset_path, fx))
            face_data.append(data_item)

            target = class_id * np.ones((data_item.shape[0],))
            labels.append(target)

            class_id += 1

    if not face_data:
        raise Exception("No dataset found in face_dataset/")

    face_dataset = np.concatenate(face_data, axis=0)
    face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

    trainset = np.concatenate((face_dataset, face_labels), axis=1)
    return trainset, names


# Load dataset pertama kali
trainset, names = load_dataset()
#############################################


# ========== FACE RECOGNITION ==========
@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected")

    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    x, y, w, h = faces[0]

    offset = 5
    face_section = img[y-offset:y+h+offset, x-offset:x+w+offset]
    face_section = cv2.resize(face_section, (100, 100))

    out = knn(trainset, face_section.flatten())
    employee_id = names[int(out)]

    return {"employee_id": employee_id}


# ========== FACE ENROLLMENT ==========
@app.post("/enroll")
async def enroll_face(
    employee_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    face_data = []

    for img_file in images:
        contents = await img_file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            continue

        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]

        offset = 5
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        face_data.append(face_section)

    if len(face_data) < MIN_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough samples ({len(face_data)}/{MIN_SAMPLES})"
        )

    face_data = np.array(face_data)
    face_data = face_data.reshape((face_data.shape[0], -1))

    os.makedirs(dataset_path, exist_ok=True)

    np.save(os.path.join(dataset_path, f"{employee_id}.npy"), face_data)

    return {"message": "Face enrolled", "samples": len(face_data), "employee_id": employee_id}


# ========== RELOAD DATASET ==========
@app.post("/reload-dataset")
async def reload_dataset():
    global trainset, names
    trainset, names = load_dataset()
    return {"message": "Dataset reloaded", "total_classes": len(names)}



@app.post("/has-face")
async def has_face(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    boxes = []
    for (x, y, w, h) in faces:
        boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    return {
        "has_face": len(faces) > 0,
        "count": len(faces),
        "boxes": boxes,  # optional, bisa dipakai nanti buat bounding box di FE
    }

# @app.post("/verify-step")
# async def verify_step(
#     step_code: str = Form(...),   # LOOK_CENTER / SMILE / TURN_LEFT / RAISE_HAND, dll
#     images: List[UploadFile] = File(...)
# ):
#     if trainset is None or len(names) == 0:
#         raise HTTPException(status_code=500, detail="Dataset not loaded")

#     recognized_ids = []

#     for img_file in images:
#         contents = await img_file.read()
#         np_arr = np.frombuffer(contents, np.uint8)
#         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#         if frame is None:
#             continue

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#         if len(faces) == 0:
#             continue

#         faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
#         x, y, w, h = faces[0]

#         offset = 5
#         face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
#         face_section = cv2.resize(face_section, (100, 100))

#         out = knn(trainset, face_section.flatten())
#         employee_id = names[int(out)]
#         recognized_ids.append(employee_id)

#         # TODO: di sini nanti bisa ditambah logika cek ekspresi/gesture

#     if not recognized_ids:
#         raise HTTPException(status_code=400, detail="No face detected in any frame")

#     # simple voting: siapa yg paling sering muncul
#     unique_ids, counts = np.unique(np.array(recognized_ids), return_counts=True)
#     best_index = np.argmax(counts)
#     best_employee_id = unique_ids[best_index]

#     # untuk sekarang: kita cuma verify ID sama, belum benar-benar cek ekspresi step_code
#     return {
#         "passed": True,
#         "employee_id": best_employee_id,
#         "step_code": step_code,
#         "frames_used": len(recognized_ids)
#     }