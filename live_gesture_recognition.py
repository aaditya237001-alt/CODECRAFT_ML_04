import cv2
import numpy as np
from keras.models import load_model

# Load the model
model = load_model('gesture_model.h5')

# Class names
class_names = ['down', 'fist', 'fist_moved', 'index', 'l', 'ok', 'palm', 'palm_moved', 'thumb', 'c']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)

    # Resize and normalize (no grayscale)
    input_img = cv2.resize(frame, (64, 64))
    input_img = input_img / 255.0
    input_img = np.reshape(input_img, (1, 64, 64, 3))  # 3 channels!

    # Predict
    prediction = model.predict(input_img)
    class_index = np.argmax(prediction)
    predicted_label = class_names[class_index]

    # Show prediction
    cv2.putText(frame, f'Gesture: {predicted_label}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.putText(frame, "Press 'q' to quit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Live Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
