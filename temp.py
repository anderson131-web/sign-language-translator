import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import Label, Frame, Canvas, Scrollbar, ttk, Button
from PIL import Image, ImageTk
import time
import os
import pygame  # type: ignore
from gtts import gTTS
from googletrans import Translator

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label dictionary (supporting A-Z)
labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z

# GUI setup
root = tk.Tk()
root.title("Sign Language Recognition")
root.geometry("900x700")
root.configure(bg="#2C3E50")

# Create a frame for the main content
container = Frame(root, bg="#2C3E50")
container.pack(fill="both", expand=True)

# Create a canvas for scrolling
canvas = Canvas(container, bg="#2C3E50", highlightthickness=0)
canvas.pack(side="left", fill="both", expand=True)

# Add a scrollbar
scrollbar = Scrollbar(container, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")
canvas.configure(yscrollcommand=scrollbar.set)

# Create a frame inside the canvas
main_frame = Frame(canvas, bg="#2C3E50")
canvas.create_window((0, 0), window=main_frame, anchor="nw")

# Function to update scrollbar region
def update_scroll_region(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

main_frame.bind("<Configure>", update_scroll_region)

recognized_sentence = ""
translated_sentence = ""
# For updating recognized_sentence, we'll use data from the first hand only.
last_predicted_character = None
character_start_time = None

label_sentence = Label(main_frame, text="Recognized Sentence: ", font=("Arial", 16, "bold"), bg="#2C3E50", fg="white")
label_sentence.pack(pady=10)

video_label = Label(main_frame, bg="black")
video_label.pack(pady=10)

# Translation label
label_translated = Label(main_frame, text="Translated Sentence: ", font=("Arial", 16, "bold"), bg="#2C3E50", fg="white")
label_translated.pack(pady=10)

# Dropdown menu for language selection
language_options = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es"
}
selected_language = tk.StringVar()
selected_language.set("English")  # Default language

dropdown = ttk.Combobox(main_frame, textvariable=selected_language, values=list(language_options.keys()), state="readonly")
dropdown.pack(pady=5)

cap = None

def start_camera():
    global cap, recognized_sentence, last_predicted_character, character_start_time
    recognized_sentence = ""
    last_predicted_character = None
    character_start_time = None
    label_sentence.config(text="Recognized Sentence: ")
    label_translated.config(text="Translated Sentence: ")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        label_sentence.config(text="Error: Camera not accessible")
        return
    process_video()

def stop_camera():
    global cap
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    video_label.config(image="")

def process_video():
    global cap, recognized_sentence, last_predicted_character, character_start_time
    if cap is None or not cap.isOpened():
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Check the number of detected hands
        if len(results.multi_hand_landmarks) == 1:
            # One hand detected: Process that hand only.
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            x_coords = [landmark.x * W for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y * H for landmark in hand_landmarks.landmark]
            x1, y1, x2, y2 = int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))
            
            # Prepare data for prediction
            data_aux = []
            min_x, min_y = min(x_coords), min(y_coords)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - (min_x / W))
                data_aux.append(landmark.y - (min_y / H))
            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict.get(int(prediction[0]), "?")
                # Update recognized sentence using hold logic
                if predicted_character == last_predicted_character:
                    if character_start_time is None:
                        character_start_time = time.time()
                    elif time.time() - character_start_time >= 1.5:
                        recognized_sentence += predicted_character
                        label_sentence.config(text="Recognized Sentence: " + recognized_sentence)
                        character_start_time = None
                else:
                    last_predicted_character = predicted_character
                    character_start_time = time.time()
            # Draw bounding box and prediction text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)
        elif len(results.multi_hand_landmarks) >= 2:
            # Two or more hands detected: Combine the landmarks to form one union bounding box.
            all_x = []
            all_y = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                x_coords = [landmark.x * W for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y * H for landmark in hand_landmarks.landmark]
                all_x.extend(x_coords)
                all_y.extend(y_coords)
            x1, y1, x2, y2 = int(min(all_x)), int(min(all_y)), int(max(all_x)), int(max(all_y))
            
            # For prediction, use the first hand's data (model expects 42 features)
            hand_landmarks = results.multi_hand_landmarks[0]
            x_coords = [landmark.x * W for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y * H for landmark in hand_landmarks.landmark]
            min_x, min_y = min(x_coords), min(y_coords)
            data_aux = []
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - (min_x / W))
                data_aux.append(landmark.y - (min_y / H))
            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict.get(int(prediction[0]), "?")
                if predicted_character == last_predicted_character:
                    if character_start_time is None:
                        character_start_time = time.time()
                    elif time.time() - character_start_time >= 1.5:
                        recognized_sentence += predicted_character
                        label_sentence.config(text="Recognized Sentence: " + recognized_sentence)
                        character_start_time = None
                else:
                    last_predicted_character = predicted_character
                    character_start_time = time.time()
            # Draw the union bounding box and display prediction
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)
    
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)
    root.after(10, process_video)

def speak_text():
    global recognized_sentence
    if recognized_sentence:
        tts = gTTS(text=recognized_sentence, lang='en')
        output_file = "output.mp3"
        
        if os.path.exists(output_file):
            os.remove(output_file)
        
        tts.save(output_file)
        pygame.mixer.init()
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        pygame.mixer.quit()
        os.remove(output_file)

def add_space():
    global recognized_sentence
    recognized_sentence += " "
    label_sentence.config(text="Recognized Sentence: " + recognized_sentence)

def clear_text(clear_all):
    global recognized_sentence
    if clear_all:
        recognized_sentence = ""
    else:
        recognized_sentence = recognized_sentence[:-1] if recognized_sentence else ""
    label_sentence.config(text="Recognized Sentence: " + recognized_sentence)

def translate_text():
    global recognized_sentence
    if recognized_sentence:
        translator = Translator()
        target_lang = language_options[selected_language.get()]
        translation = translator.translate(recognized_sentence, dest=target_lang)
        label_translated.config(text="Translated Sentence: " + translation.text)

# Buttons
btn_start = tk.Button(main_frame, text="Start Camera", command=start_camera, font=("Arial", 14, "bold"), 
                      bg="#27AE60", fg="white", padx=20, pady=5)
btn_start.pack(pady=10)

btn_stop = tk.Button(main_frame, text="Stop Camera", command=stop_camera, font=("Arial", 14, "bold"), 
                     bg="#E74C3C", fg="white", padx=20, pady=5)
btn_stop.pack(pady=5)

btn_speak = tk.Button(main_frame, text="Speak", command=speak_text, font=("Arial", 14, "bold"), 
                      bg="#2980B9", fg="white", padx=20, pady=5)
btn_speak.pack(pady=5)

btn_translate = tk.Button(main_frame, text="Translate", command=translate_text, font=("Arial", 14, "bold"),
                          bg="#8E44AD", fg="white", padx=20, pady=5)
btn_translate.pack(pady=5)

btn_space = tk.Button(main_frame, text="Space", command=add_space, font=("Arial", 14, "bold"), 
                      bg="#F39C12", fg="white", padx=20, pady=5)
btn_space.pack(pady=5)

btn_clear_all = tk.Button(main_frame, text="Clear All", command=lambda: clear_text(True), font=("Arial", 14, "bold"), 
                          bg="#C0392B", fg="white", padx=20, pady=5)
btn_clear_all.pack(pady=5)

btn_clear_letter = tk.Button(main_frame, text="Clear Letter", command=lambda: clear_text(False), font=("Arial", 14, "bold"), 
                             bg="#D35400", fg="white", padx=20, pady=5)
btn_clear_letter.pack(pady=5)

root.mainloop()
