import pickle
import cv2
import numpy as np

data = {
   0:'angry', 1:'disgusted', 2:'fearful', 3:'happy', 4:'neutral', 5:'sad', 6:'surprised' 
}

# Загрузка модели
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Запуск камеры
cam = cv2.VideoCapture(0)

IMG_SIZE = 48 # должен совпадать с тем, что использовалось при обучении

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Копия кадра для отображения
    display_frame = frame.copy()

    # Предобработка изображения
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # если модель обучалась на grayscale

    # Нормализация
    img = img / 255.0

    # Преобразование в вектор
    img = img.reshape(1, 48, 48, 1)

    pred = np.argmax(model.predict(img))

    text = data[pred]
    color = (0, 0, 255)

    cv2.putText(display_frame, text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Human Classification", display_frame)

    # Выход по ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()