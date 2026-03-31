# import cv2
# import pickle

# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("face_model.xml")

# with open("labels_lbph.pkl", "rb") as f:
#     labels = pickle.load(f)

# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# )

# cap = cv2.VideoCapture(0)

# print("Камера запущена...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:

#         face = gray[y:y+h, x:x+w]
#         face = cv2.resize(face, (200, 200))

#         label, confidence = recognizer.predict(face)

#         if confidence < 80:
#             name = labels[label]
#         else:
#             name = "Unknown"

#         text = f"{name} ({confidence:.2f})"

#         cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
#         cv2.putText(frame, text, (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                     (0,255,0), 2)

#     cv2.imshow("Face Recognition LBPH", frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# 🔥 загрузка модели
model = load_model("model.keras")

# 🔥 загрузка labels
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# переворачиваем словарь
label_map = {v: k for k, v in labels.items()}

# 🔥 детектор лица
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

print("Камера запущена...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # 🔥 вырезаем лицо
        face = frame[y:y+h, x:x+w]

        # 🔥 ТОЧНО ТАК ЖЕ как в train
        face = cv2.resize(face, (224, 224))

        # ❗ ВАЖНО: у тебя НЕ было нормализации в train
        # поэтому НЕ делаем /255.0

        face = np.expand_dims(face, axis=0)

        # 🔥 предсказание
        prediction = model.predict(face, verbose=0)[0]

        class_id = np.argmax(prediction)
        confidence = np.max(prediction)

        name = label_map[class_id]

        # 🔥 порог
        if confidence < 0.35:
            name = "Unknown"

        text = f"{name} ({confidence:.2f})"

        # 🔥 рисуем
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0), 2)

    cv2.imshow("Face Recognition (TensorFlow)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()