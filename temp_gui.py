import cv2
import mediapipe as mp
import numpy as np
import pickle
import tkinter as tk
from PIL import Image, ImageTk
import threading

# Loading the trained model
model_dict = pickle.load(open('D:\\SignLanguageProject\\SILT\\model.p','rb'))
model = model_dict['model']

# Function to capture webcam feed
def capture_feed():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    def process_frame():
        while True:
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:

                        prediction = model.predict([np.asarray([landmark.x, landmark.y] * 42)])
                        print("Predicted label:", prediction[0])

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(img)
            photo = ImageTk.PhotoImage(image=img)
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    threading.Thread(target=process_frame, daemon=True).start()

root = tk.Tk()
root.title("Sign Language Recognition")
root.geometry("800x600")

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

capture_feed()

root.mainloop()
