import cv2
import numpy as np
import mediapipe as mp
import os

# ==========================
# FACE KNN FUNCTIONS
# ==========================
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
    return output[0][np.argmax(output[1])]


# ==========================
# LOAD FACE DATASET
# ==========================
dataset_path = "./face_dataset/"
face_data = []
labels = []
names = {}
class_id = 0

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data = np.load(dataset_path + fx)
        face_data.append(data)
        labels.append(class_id * np.ones((data.shape[0],)))
        class_id += 1

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
trainset = np.concatenate((face_dataset, face_labels), axis=1)


# ==========================
# MEDIAPIPE HANDS
# ==========================
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def detect_gesture(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)

    if not result.multi_hand_landmarks:
        return []

    gestures = []

    for lm, hd in zip(result.multi_hand_landmarks, result.multi_handedness):

        hand_label = hd.classification[0].label  # "Left" / "Right"

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

        if (ujung_jempol.y < ujung_telunjuk.y and
            ujung_jempol.y < ujung_tengah.y and
            ujung_jempol.y < ujung_manis.y and
            ujung_jempol.y < ujung_kelingking.y):
            gesture = "Thumbs Up"

        elif (ujung_telunjuk.y < pangkal_telunjuk.y and
              ujung_tengah.y < pangkal_tengah.y and
              ujung_manis.y > pangkal_manis.y and
              ujung_kelingking.y > pangkal_kelingking.y):
            gesture = "Peace"

        elif (ujung_telunjuk.y < pangkal_telunjuk.y and
              ujung_kelingking.y < pangkal_kelingking.y and
              ujung_tengah.y > pangkal_tengah.y and
              ujung_manis.y > pangkal_manis.y):
            gesture = "Rock"

        elif (abs(ujung_jempol.x - ujung_telunjuk.x) < 0.05 and
              abs(ujung_jempol.y - ujung_telunjuk.y) < 0.05 and
              ujung_tengah.y < pangkal_tengah.y and
              ujung_manis.y < pangkal_manis.y and
              ujung_kelingking.y < pangkal_kelingking.y):
            gesture = "OK"

        if gesture:
            gestures.append({"hand": hand_label, "gesture": gesture})

    return gestures


# ==========================
# MAIN LOOP
# ==========================
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # ============= MIRROR =============
    frame = cv2.flip(frame, 1)
    # ==================================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # GESTURE DETECT
    gestures = detect_gesture(frame)

    # TAMPILKAN GESTURE
    y_offset = 30
    for g in gestures:
        cv2.putText(frame, f"{g['hand']}: {g['gesture']}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        y_offset += 30

    # FACE DETECT + RECOGNIZE
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_section = frame[y:y+h, x:x+w]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())
        name = names[int(out)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

    cv2.imshow("Face + Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
