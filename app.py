from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
import numpy as np
import cv2
import os
import mediapipe as mp
from datetime import datetime

app = FastAPI()

# ==============================
# CONFIGURATIONS
# ==============================
cascade_path = "haarcascade_frontalface_alt.xml"
if not os.path.exists(cascade_path):
    # Fallback ke cascade bawaan OpenCV
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    print(f"Using fallback cascade: {cascade_path}")

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise Exception("Failed to load face cascade classifier!")

dataset_path = "./face_dataset/"
MIN_SAMPLES = 20
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

# ==============================
# KNN FUNCTIONS
# ==============================
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


# ==============================
# LOAD DATASET
# ==============================
def load_dataset():
    face_data = []
    labels = []
    class_id = 0
    names = {}

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
        raise Exception("Dataset folder created but empty. Please enroll faces first.")

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


# Initialize dataset (handle empty dataset gracefully)
try:
    trainset, names = load_dataset()
except Exception as e:
    print(f"Warning: {e}")
    trainset, names = np.array([]), {}


# ==============================
# HELPER: Validate File
# ==============================
def validate_file(file: UploadFile):
    """Validate file type and size"""
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large (max {MAX_FILE_SIZE//1024//1024}MB)")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}")


# ==============================
# HELPER: Decode Image
# ==============================
async def decode_image(file: UploadFile):
    """Safely decode uploaded image"""
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail=f"Cannot decode image: {file.filename}")
        
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")


# ==============================
# GESTURE DETECTION (Backend Only)
# ==============================
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
ALLOWED_GESTURES = ["Thumbs Up", "Peace", "Rock", "OK"]


def detect_gesture(frame):
    """
    Detect hand gestures in a frame.
    This is a backend function, not exposed as API endpoint.
    Can be called internally by other backend services.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)

    if not result.multi_hand_landmarks or not result.multi_handedness:
        return []

    gestures = []
    for lm, hd in zip(result.multi_hand_landmarks, result.multi_handedness):
        hand_label = hd.classification[0].label

        ujung_jempol = lm.landmark[mp_hands.HandLandmark.THUMB_TIP]
        ujung_telunjuk = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        ujung_tengah = lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ujung_manis = lm.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        ujung_kelingking = lm.landmark[mp_hands.HandLandmark.PINKY_TIP]

        pangkal_telunjuk = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pangkal_tengah = lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        pangkal_manis = lm.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
        pangkal_kelingking = lm.landmark[mp_hands.HandLandmark.PINKY_MCP]

        gesture = None

        # Thumbs Up
        if (ujung_jempol.y < ujung_telunjuk.y and
            ujung_jempol.y < ujung_tengah.y and
            ujung_jempol.y < ujung_manis.y and
            ujung_jempol.y < ujung_kelingking.y):
            gesture = "Thumbs Up"

        # Peace
        elif (ujung_telunjuk.y < pangkal_telunjuk.y and
              ujung_tengah.y < pangkal_tengah.y and
              ujung_manis.y > pangkal_manis.y and
              ujung_kelingking.y > pangkal_kelingking.y):
            gesture = "Peace"

        # Rock
        elif (ujung_telunjuk.y < pangkal_telunjuk.y and
              ujung_kelingking.y < pangkal_kelingking.y and
              ujung_tengah.y > pangkal_tengah.y and
              ujung_manis.y > pangkal_manis.y):
            gesture = "Rock"

        # OK
        elif (abs(ujung_jempol.x - ujung_telunjuk.x) < 0.05 and
              abs(ujung_jempol.y - ujung_telunjuk.y) < 0.05 and
              ujung_tengah.y < pangkal_tengah.y and
              ujung_manis.y < pangkal_manis.y and
              ujung_kelingking.y < pangkal_kelingking.y):
            gesture = "OK"

        if gesture:
            gestures.append({"hand": hand_label, "gesture": gesture})

    return gestures


def validate_gesture_level(frames, level="basic"):
    """
    Validate gesture requirements for enrollment levels.
    Backend function to check if frames meet level criteria.
    
    Args:
        frames: List of numpy arrays (images)
        level: "basic", "pro", or "ultra"
    
    Returns:
        dict with validation results
    """
    detected_gestures = []
    frames_with_two_hands = []
    
    for frame in frames:
        gestures = detect_gesture(frame)
        if gestures:
            detected_gestures.extend(gestures)
            
            hands = {g["hand"]: g["gesture"] for g in gestures}
            if "Left" in hands and "Right" in hands:
                frames_with_two_hands.append(hands)
    
    result = {
        "valid": False,
        "level": level,
        "detected_gestures": detected_gestures,
        "error": None
    }
    
    # BASIC - no gesture required
    if level == "basic":
        result["valid"] = True
        return result
    
    # PRO - at least one gesture
    elif level == "pro":
        valid_gesture = any(g["gesture"] in ALLOWED_GESTURES for g in detected_gestures)
        if not valid_gesture:
            result["error"] = f"PRO: Requires at least one gesture from {ALLOWED_GESTURES}"
            return result
        result["valid"] = True
        return result
    
    # ULTRA - both hands with different gestures
    elif level == "ultra":
        if len(frames_with_two_hands) == 0:
            result["error"] = "ULTRA: Requires both left & right hand in same frame"
            return result
        
        valid_ultra = False
        valid_combination = None
        for hands in frames_with_two_hands:
            L = hands.get("Left")
            R = hands.get("Right")
            if L in ALLOWED_GESTURES and R in ALLOWED_GESTURES and L != R:
                valid_ultra = True
                valid_combination = f"Left: {L}, Right: {R}"
                break
        
        if not valid_ultra:
            result["error"] = f"ULTRA: Requires DIFFERENT gestures from {ALLOWED_GESTURES}"
            return result
        
        result["valid"] = True
        result["valid_combination"] = valid_combination
        result["two_hand_frames"] = len(frames_with_two_hands)
        return result
    
    result["error"] = "Invalid level"
    return result


# ==============================
# API ENDPOINTS
# ==============================

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "running",
        "enrolled_employees": len(names),
        "dataset_loaded": len(trainset) > 0
    }


@app.get("/employees")
async def list_employees():
    """List all enrolled employees"""
    return {
        "total": len(names),
        "employees": list(names.values())
    }


@app.post("/has-face")
async def has_face(file: UploadFile = File(...)):
    """Check if image contains face(s)"""
    validate_file(file)
    img = await decode_image(file)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    boxes = []
    for (x, y, w, h) in faces:
        boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    return {
        "has_face": len(faces) > 0,
        "count": len(faces),
        "boxes": boxes
    }


@app.post("/enroll")
async def enroll_face(
    employee_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Enroll a new employee with face images.
    Requires at least 20 valid face images.
    """
    if not employee_id or len(employee_id.strip()) == 0:
        raise HTTPException(status_code=400, detail="employee_id cannot be empty")
    
    # Check if already exists
    employee_file = os.path.join(dataset_path, f"{employee_id}.npy")
    if os.path.exists(employee_file):
        raise HTTPException(status_code=400, detail=f"Employee {employee_id} already enrolled. Use different ID.")
    
    face_data = []

    for img_file in files:
        validate_file(img_file)
        
        try:
            frame = await decode_image(img_file)
        except:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            continue

        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]

        offset = 5
        y1 = max(0, y - offset)
        y2 = min(frame.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(frame.shape[1], x + w + offset)
        
        face_section = frame[y1:y2, x1:x2]
        face_section = cv2.resize(face_section, (100, 100))

        face_data.append(face_section)

    if len(face_data) < MIN_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough valid face samples ({len(face_data)}/{MIN_SAMPLES}). Upload more images with clear faces."
        )

    face_data = np.array(face_data)
    face_data = face_data.reshape((face_data.shape[0], -1))

    os.makedirs(dataset_path, exist_ok=True)
    np.save(employee_file, face_data)

    # Auto reload dataset
    global trainset, names
    trainset, names = load_dataset()

    return {
        "message": "Face enrolled successfully",
        "samples": len(face_data),
        "employee_id": employee_id,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """
    Recognize a face from an image.
    Returns employee_id if face is recognized.
    """
    if len(names) == 0:
        raise HTTPException(status_code=400, detail="No enrolled faces in database")
    
    validate_file(file)
    img = await decode_image(file)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected")

    # Get largest face
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]

    offset = 5
    y1 = max(0, y - offset)
    y2 = min(img.shape[0], y + h + offset)
    x1 = max(0, x - offset)
    x2 = min(img.shape[1], x + w + offset)
    
    face_section = img[y1:y2, x1:x2]
    face_section = cv2.resize(face_section, (100, 100))

    out = knn(trainset, face_section.flatten())
    employee_id = names[int(out)]

    return {
        "employee_id": employee_id,
        "confidence": "high",
        "face_location": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    }


@app.post("/reload-dataset")
async def reload_dataset():
    """Reload the face dataset from disk"""
    global trainset, names
    try:
        trainset, names = load_dataset()
        return {
            "message": "Dataset reloaded successfully",
            "total_employees": len(names),
            "employee_ids": list(names.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))