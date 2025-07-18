import os
import pickle
import mediapipe as mp
import cv2
import numpy as np


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './sign lanuage update/sign-language-detector-python/data'

data = []
labels = []


EXPECTED_LANDMARKS = 21
COORDS_PER_LANDMARK = 2
EXPECTED_FEATURES = EXPECTED_LANDMARKS * COORDS_PER_LANDMARK

for dir_ in os.listdir(DATA_DIR):
    
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue
        
    print(f"Processing class: {dir_}")
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        
        if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        
        data_aux = []
        
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            print(f"Could not read image: {os.path.join(DATA_DIR, dir_, img_path)}")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Collect all x,y coordinates first to normalize
                x_ = []
                y_ = []
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                
                
                if not x_ or not y_:
                    continue
                    
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
            
            
            if len(data_aux) == EXPECTED_FEATURES:
                data.append(data_aux)
                labels.append(dir_)
            else:
                
                if len(data_aux) < EXPECTED_FEATURES:
                    
                    data_aux.extend([0] * (EXPECTED_FEATURES - len(data_aux)))
                else:
                    
                    data_aux = data_aux[:EXPECTED_FEATURES]
                
                data.append(data_aux)
                labels.append(dir_)
        else:
           
            data.append([0] * EXPECTED_FEATURES)
            labels.append(dir_)


data_np = np.array(data)
labels_np = np.array(labels)

print(f"Dataset created with shape: {data_np.shape}")
print(f"Number of classes: {len(set(labels_np))}")
print(f"Number of samples per class: {np.unique(labels_np, return_counts=True)}")

# Save the dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset saved to data.pickle")