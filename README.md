# CODECRAFT_ML_04
# Hand Gesture Recognition with Real-Time Control System

This project implements a hand gesture recognition model using deep learning and computer vision to classify hand gestures from image and video data. It also includes a real-time system that captures webcam input and maps recognized gestures to intuitive human-computer interaction or control actions.

---

## 🧠 Project Goals

- Build a CNN-based model to classify 10 hand gestures
- Train and evaluate the model on a custom gesture dataset
- Enable real-time gesture recognition using a webcam
- Integrate gesture-based control (e.g., simulate key presses)

---

## 📁 Dataset

- Source: [leapGestRecog Dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
- Categories: `'palm'`, `'fist'`, `'thumb'`, `'ok'`, `'l'`, `'c'`, `'index'`, `'down'`, `'fist_moved'`, `'palm_moved'`
- Preprocessing: Flattened folder structure, grayscale to RGB conversion, resized to `64x64`.

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib
- PyAutoGUI (for control system)
- Google Colab (for training)
- VS Code / local system (for real-time recognition)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition
2. Install requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Train the model (Optional)
Use Colab to run the training notebook (train_model.ipynb).
Make sure dataset is uploaded to /content/hand_gesture_dataset.

Or, skip training and use the pre-trained model gesture_model.h5.

🎥 Live Gesture Recognition
Run this Python file on your local machine:

bash
Copy
Edit
python live_gesture_recognition.py
It uses your webcam to detect hand gestures and maps them to controls.
E.g., shows labels like “Fist”, “Thumb”, and can trigger keypress actions if configured.

📂 File Structure
bash
Copy
Edit
hand-gesture-recognition/
│
├── final_dataset/              # Processed dataset (after flattening)
├── train_model.ipynb           # Model training (Colab)
├── gesture_model.h5            # Trained model
├── live_gesture_recognition.py # Real-time webcam detection
├── requirements.txt            # Python dependencies
└── README.md                   # This file
