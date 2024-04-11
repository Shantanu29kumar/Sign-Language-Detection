# import cv2
# import mediapipe as mp
#
#
# # 0-> primary camera
# cap = cv2. VideoCapture(0)
#
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#
# while True:
#     ret, frame = cap.read()
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame_rgb,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style()
#             )
#
#
#
#     cv2.imshow('frame',frame)
#     cv2.waitKey(25)
#
# cap.release()
# cv2.destroyAllWindows()


# import pickle
# import numpy as np
# import cv2
# import mediapipe as mp
# import numpy as np
#
# model_dict = pickle.load(open('D:\\SignLanguageProject\\SILT\\model.p','rb'))
# model = model_dict['model']
#
# # 0 -> primary camera
# cap = cv2.VideoCapture(0)
#
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#
# while True:
#     data_aux = []
#     ret, frame = cap.read()
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style()
#             )
#
#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 # z = hand_landmarks.landmark[i].z
#                 data_aux.append(x)
#                 data_aux.append(y)
#         model.predict([np.asarray(data_aux)])
#
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
#
# # Load the trained model
# model_dict = pickle.load(open('model.p', 'rb'))
# model = model_dict['model']
#
# # 0 -> primary camera
# cap = cv2.VideoCapture(0)
#
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#
# labels_dict = {
#     0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
#     5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
#     10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
#     15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
#     20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
#     25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
#     30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
#     35: 'Z'
# }
#
# while True:
#     data_aux = []
#     ret, frame = cap.read()
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style()
#             )
#
#         for hand_landmarks in results.multi_hand_landmarks:
#             for landmark in hand_landmarks.landmark:
#                 # Append (x, y) coordinates four times
#                 data_aux.append(landmark.x)
#                 data_aux.append(landmark.y)
#                 data_aux.append(landmark.x)
#                 data_aux.append(landmark.y)
#
#     if len(data_aux) == 84:  # Ensure we have exactly 84 features
#         # Reshape the data for prediction
#         data_aux = np.asarray(data_aux).reshape(1, -1)
#
#         # Make prediction
#         prediction = model.predict(data_aux)
#
#         # Map prediction to label
#         predicted_character = labels_dict.get(prediction[0], "Unknown")
#         print("Predicted character:", predicted_character)
#
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
import pickle

model_dict = pickle.load(open('D:\\SignLanguageProject\\SILT\\model.p','rb'))
model = model_dict['model']

# 0 -> primary camera
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H,W,_ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for landmark in hand_landmarks.landmark:
                # Append (x, y) coordinates four times
                x_.append(landmark.x)
                y_.append(landmark.y)

        if x_ and y_:
            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)

            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)

            if len(x_) == len(y_) == 21:  # Ensure we have exactly 21 landmarks
                for landmark in hand_landmarks.landmark:
                    # Append (x, y) coordinates four times
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)

                # Make prediction if data_aux has exactly 84 features
                if len(data_aux) == 84:
                    prediction = model.predict([np.asarray(data_aux)])
                    print("Predicted label:", prediction)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, str(prediction[0]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
