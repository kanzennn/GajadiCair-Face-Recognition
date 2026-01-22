import cv2
import numpy as np
import os
import mediapipe as mp
from gesture_dataset.thumbsup import isThumbsup
from gesture_dataset.peace import isPeace
from gesture_dataset.rock import isRock
from gesture_dataset.ok import isOk
from gesture_dataset.l import isL
from gesture_dataset.fist import isFist
from gesture_dataset.hi import isHi
from gesture_dataset.three import isThree
from gesture_dataset.pointing import isPointing


cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

dataset_path = "./face_dataset/"

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
            labels.append(class_id * np.ones((data_item.shape[0],)))
            class_id += 1

    face_dataset = np.concatenate(face_data, axis=0)
    face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
    trainset = np.concatenate((face_dataset, face_labels), axis=1)

    return trainset, names

trainset, names = load_dataset()
print("Loaded employees:", names)

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)

GESTURE_REGISTRY = [
    ("OK", isOk),
    ("L", isL),
    ("Peace", isPeace),
    ("Rock", isRock),
    ("Three", isThree),
    ("Pointing", isPointing),
    ("Thumbs-Up", isThumbsup),
    ("Hi", isHi),
    ("Fist", isFist)
]

def detect_gesture(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)
    gestures = []

    if not result.multi_hand_landmarks:
        return gestures

    for i, lm in enumerate(result.multi_hand_landmarks):

        hand_label = "Unknown"
        if result.multi_handedness:
            hand_label = result.multi_handedness[i].classification[0].label

        ujung_jempol     = lm.landmark[mp_hands.HandLandmark.THUMB_TIP]
        ujung_telunjuk   = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        ujung_tengah     = lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ujung_manis      = lm.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        ujung_kelingking = lm.landmark[mp_hands.HandLandmark.PINKY_TIP]

        pangkal_telunjuk   = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pangkal_tengah     = lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        pangkal_manis      = lm.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
        pangkal_kelingking = lm.landmark[mp_hands.HandLandmark.PINKY_MCP]
        pangkal_jempol     = lm.landmark[mp_hands.HandLandmark.THUMB_IP]

        for name, func in GESTURE_REGISTRY:
            if func(
                ujung_jempol,
                ujung_telunjuk,
                ujung_tengah,
                ujung_manis,
                ujung_kelingking,
                pangkal_jempol,
                pangkal_telunjuk,
                pangkal_tengah,
                pangkal_manis,
                pangkal_kelingking
            ):
                gestures.append(f"{hand_label}: {name}")
                break

    return gestures

cap = cv2.VideoCapture(0)

print("Press Q to exit")
cv2.namedWindow("Face & Gesture Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face & Gesture Recognition", 1000, 700)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face_flat = face.reshape(1, -1)

        out = knn(trainset, face_flat[0])
        name = names[int(out)]

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    gestures = detect_gesture(frame)
    for i, g in enumerate(gestures):
        cv2.putText(frame, g, (10, 30 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.imshow("Face & Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
