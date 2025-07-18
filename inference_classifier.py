import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import Label, Frame, ttk, Button, simpledialog, messagebox
from PIL import Image, ImageTk
import time
import os
import threading
import pyttsx3


model_dict = pickle.load(open('./sign lanuage update/sign-language-detector-python/model.p', 'rb'))
model = model_dict['model']


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)


labels_dict = {i: chr(65 + i) for i in range(26)}
labels_dict.update({
    26: "THANK YOU",
    27: "HELLO",
    28: "I LOVE YOU",
    29: "YES",
    30: "HELP"
})

offline_translations = {
    "English": {
        "hello": {
            "Spanish": "hola", 
            "French": "bonjour", 
            "German": "hallo", 
            "Italian": "ciao",
            "Japanese": "こんにちは",
            "Chinese": "你好",
            "Portuguese": "olá",
            "Arabic": "مرحبا",
            "Russian": "привет",
            "Malayalam": "നമസ്കാരം"  
        },
        "thank you": {
            "Spanish": "gracias", 
            "French": "merci", 
            "German": "danke", 
            "Italian": "grazie",
            "Japanese": "ありがとう",
            "Chinese": "谢谢",
            "Portuguese": "obrigado",
            "Arabic": "شكرا",
            "Russian": "спасибо",
            "Malayalam": "നന്ദി"  
        },
        "i love you": {
            "Spanish": "te quiero", 
            "French": "je t'aime", 
            "German": "ich liebe dich", 
            "Italian": "ti amo",
            "Japanese": "愛してる",
            "Chinese": "我爱你",
            "Portuguese": "eu te amo",
            "Arabic": "أحبك",
            "Russian": "я люблю тебя",
            "Malayalam": "ഞാൻ നിന്നെ സ്നേഹിക്കുന്നു"  
        },
        "yes": {
            "Spanish": "sí", 
            "French": "oui", 
            "German": "ja", 
            "Italian": "sì",
            "Japanese": "はい",
            "Chinese": "是的",
            "Portuguese": "sim",
            "Arabic": "نعم",
            "Russian": "да",
            "Malayalam": "അതെ"  
        },
        "help": {
            "Spanish": "ayuda", 
            "French": "aide", 
            "German": "hilfe", 
            "Italian": "aiuto",
            "Japanese": "助けて",
            "Chinese": "帮助",
            "Portuguese": "ajuda",
            "Arabic": "مساعدة",
            "Russian": "помощь",
            "Malayalam": "സഹായം"  
        }
    }
}

class IPCameraDialog(simpledialog.Dialog):
    def body(self, master):
        tk.Label(master, text="IP Webcam Camera Configuration", font=("Arial", 14)).grid(row=0, columnspan=2, pady=10)
        tk.Label(master, text="Enter the full IP Webcam URL (e.g., http://192.168.1.xxx:8080/video)", 
                 font=("Arial", 10)).grid(row=1, columnspan=2, pady=5)
        
        tk.Label(master, text="Camera URL:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        self.url_entry = tk.Entry(master, width=40)
        self.url_entry.grid(row=2, column=1, padx=5, pady=5)
        self.url_entry.insert(0, "http://192.168.1.xxx:8080/video")
        
        return self.url_entry

    def apply(self):
        url = self.url_entry.get().strip()
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                self.result = url
                cap.release()
            else:
                messagebox.showerror("Connection Error", "Unable to connect to the camera. Please check the URL.")
                self.result = None
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.result = None

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator")
        self.root.geometry("800x750")
        self.root.configure(bg="#2C3E50")
        
        self.ip_camera_url = self.prompt_ip_camera_url()
        
        # Initialize variables
        self.cap = None
        self.recognized_sentence = ""
        self.last_predicted_character = None
        self.character_start_time = None
        self.is_running = False
        self.thread = None
        
        self.camera_source = "ip" if self.ip_camera_url else "local"
        
        # Language options
        self.language_options = {
            "English": "en", 
            "Spanish": "es", 
            "French": "fr", 
            "German": "de", 
            "Italian": "it", 
            "Japanese": "ja", 
            "Chinese": "zh-cn",
            "Portuguese": "pt",
            "Arabic": "ar",
            "Russian": "ru",
            "Malayalam":"ml"
        }
        
        # Initialize TTS engine
        self.engine = pyttsx3.init()
        
        # Initialize mode
        self.offline_mode = tk.BooleanVar()
        self.offline_mode.set(False)
        
        self.setup_ui()
    
    def prompt_ip_camera_url(self):
        use_ip = messagebox.askyesno("Camera Source", 
                                     "Do you want to use an IP Camera?\n\n"
                                     "Click 'Yes' to enter IP Camera URL\n"
                                     "Click 'No' to use local camera")
        
        if use_ip:
            dialog = IPCameraDialog(self.root, title="IP Camera Configuration")
            return dialog.result
        
        return None

    def setup_ui(self):
       
        main_frame = Frame(self.root, bg="#2C3E50")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        
        self.status_frame = Frame(main_frame, bg="#2C3E50")
        self.status_frame.pack(fill="x", pady=5)
        
        self.status_indicator = Label(self.status_frame, text="•", font=("Arial", 24), fg="#E74C3C", bg="#2C3E50")
        self.status_indicator.pack(side="left", padx=5)
        
        self.status_text = Label(self.status_frame, text="Camera Off", font=("Arial", 14), fg="white", bg="#2C3E50")
        self.status_text.pack(side="left")
        
       
        self.connection_status = Label(self.status_frame, text=f"Source: {'IP Camera' if self.camera_source == 'ip' else 'Local Camera'}", 
                                       font=("Arial", 12), fg="#F39C12", bg="#2C3E50")
        self.connection_status.pack(side="left", padx=10)
        
       
        offline_frame = Frame(self.status_frame, bg="#2C3E50")
        offline_frame.pack(side="right", padx=10)
        
        offline_checkbox = ttk.Checkbutton(offline_frame, text="Offline Mode", 
                                          variable=self.offline_mode, 
                                          command=self.toggle_offline_mode)
        offline_checkbox.pack(side="right")
        
       
        self.video_frame = Frame(main_frame, bg="black", width=640, height=480)
        self.video_frame.pack(pady=10)
        self.video_frame.pack_propagate(False)
        
        self.video_label = Label(self.video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)
        
        
        text_frame = Frame(main_frame, bg="#2C3E50")
        text_frame.pack(fill="x", pady=10)
        
        self.label_sentence = Label(text_frame, text="Recognized: ", font=("Arial", 16), bg="#2C3E50", fg="white", anchor="w")
        self.label_sentence.pack(fill="x")
        
        self.label_translated = Label(text_frame, text="Translated: ", font=("Arial", 16), bg="#2C3E50", fg="white", anchor="w")
        self.label_translated.pack(fill="x", pady=5)
        
        
        control_frame = Frame(main_frame, bg="#2C3E50")
        control_frame.pack(fill="x", pady=10)
        
       
        language_frame = Frame(control_frame, bg="#2C3E50")
        language_frame.pack(side="top", fill="x", pady=5)
        
        Label(language_frame, text="Translate to:", font=("Arial", 12), bg="#2C3E50", fg="white").pack(side="left", padx=5)
        
        self.selected_language = tk.StringVar()
        self.selected_language.set("Spanish")
        
        style = ttk.Style()
        style.configure("TCombobox", foreground="#2C3E50", background="#ECF0F1", font=("Arial", 12))
        
        dropdown = ttk.Combobox(language_frame, textvariable=self.selected_language, 
                              values=list(self.language_options.keys()), state="readonly", width=15)
        dropdown.pack(side="left", padx=5)
        
       
        self.button_frame = Frame(main_frame, bg="#2C3E50")
        self.button_frame.pack(fill="x", pady=10)
        
      
        row1 = Frame(self.button_frame, bg="#2C3E50")
        row1.pack(fill="x", pady=5)
        
        self.btn_start = Button(row1, text="Start Camera", command=self.toggle_camera, 
                             font=("Arial", 12, "bold"), bg="#27AE60", fg="white", width=15)
        self.btn_start.pack(side="left", padx=5)
        
        Button(row1, text="Space", command=self.add_space, 
               font=("Arial", 12, "bold"), bg="#F39C12", fg="white", width=10).pack(side="left", padx=5)
        
        Button(row1, text="Delete", command=lambda: self.clear_text(False), 
               font=("Arial", 12, "bold"), bg="#D35400", fg="white", width=10).pack(side="left", padx=5)
        
        Button(row1, text="Clear All", command=lambda: self.clear_text(True), 
               font=("Arial", 12, "bold"), bg="#C0392B", fg="white", width=10).pack(side="left", padx=5)
        
        self.speak_btn = Button(row1, text="Speak Recognized", command=lambda: threading.Thread(target=self.speak_text).start(), 
               font=("Arial", 12, "bold"), bg="#2980B9", fg="white", width=15)
        self.speak_btn.pack(side="left", padx=5)
        
        self.translate_btn = Button(row1, text="Translate", command=lambda: threading.Thread(target=self.translate_text).start(), 
               font=("Arial", 12, "bold"), bg="#8E44AD", fg="white", width=15)
        self.translate_btn.pack(side="left", padx=5)
        
        
        self.ip_camera_btn = Button(row1, text="Reconfigure IP Camera", 
               command=self.reconfigure_ip_camera, 
               font=("Arial", 12, "bold"), bg="#34495E", fg="white", width=20)
        self.ip_camera_btn.pack(side="left", padx=5)
        
        # Help and instructions
        tip_frame = Frame(main_frame, bg="#34495E", bd=1, relief="solid")
        tip_frame.pack(fill="x", pady=10)
        
        tip_text = """
        ✓ Hold your hand sign still for 1.5 seconds to record a letter
        ✓ Use space between words
        ✓ Start with simple signs and add to your sentence
        ✓ Check 'Offline Mode' to use without internet connection
        """
        Label(tip_frame, text=tip_text, font=("Arial", 11), bg="#34495E", fg="white", 
              justify="left", padx=10, pady=5).pack(fill="x")
        
        
        self.offline_mode_label = Label(main_frame, 
                                      text="OFFLINE MODE: Limited translations available", 
                                      font=("Arial", 12, "bold"), 
                                      bg="#E67E22", fg="white", 
                                      pady=5)
       
        if self.offline_mode.get():
            self.offline_mode_label.pack(fill="x", pady=5)

    def toggle_offline_mode(self):
        if self.offline_mode.get():
            self.offline_mode_label.pack(fill="x", pady=5)
            messagebox.showinfo("Offline Mode", 
                              "Offline mode enabled. Only basic translations available.\n"
                              "Full phrases may not translate correctly.")
        else:
            self.offline_mode_label.pack_forget()

    def reconfigure_ip_camera(self):
        dialog = IPCameraDialog(self.root, title="Reconfigure IP Camera")
        if dialog.result:
            self.ip_camera_url = dialog.result
            self.camera_source = "ip"
            self.connection_status.config(text=f"IP: {self.ip_camera_url}", fg="#F39C12")
            if self.is_running:
                self.stop_camera()

    def start_camera(self):
        self.recognized_sentence = ""
        self.last_predicted_character = None
        self.character_start_time = None
        self.label_sentence.config(text="Recognized: ")
        self.label_translated.config(text="Translated: ")
        
        try:
            if self.camera_source == "local":
                self.cap = cv2.VideoCapture(0)
            elif self.camera_source == "ip":
                if not self.ip_camera_url:
                    self.ip_camera_url = self.prompt_ip_camera_url()
                
                if not self.ip_camera_url:
                    messagebox.showerror("Error", "No IP Camera URL provided")
                    return
                
                self.cap = cv2.VideoCapture(self.ip_camera_url)
            
            if not self.cap.isOpened():
                self.label_sentence.config(text="Error: Camera not accessible")
                return
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_running = True
            self.status_indicator.config(fg="#27AE60")
            self.status_text.config(text="Camera Active")
            
            self.thread = threading.Thread(target=self.process_video)
            self.thread.daemon = True
            self.thread.start()
        except Exception as e:
            self.label_sentence.config(text=f"Error: {str(e)}")

    def toggle_camera(self):
        if self.is_running:
            self.stop_camera()
            self.btn_start.config(text="Start Camera", bg="#27AE60")
        else:
            self.start_camera()
            self.btn_start.config(text="Stop Camera", bg="#E74C3C")

    def stop_camera(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.video_label.config(image="")
        self.status_indicator.config(fg="#E74C3C")
        self.status_text.config(text="Camera Off")

    def process_video(self):
        prev_time = time.time()
        frame_count = 0
        fps = 0
        
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 2 != 0:
                continue
                
            current_time = time.time()
            if current_time - prev_time >= 1:
                fps = frame_count
                frame_count = 0
                prev_time = current_time
            
            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(frame_rgb)
            
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                x_coords = [landmark.x * W for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y * H for landmark in hand_landmarks.landmark]
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                
                data_aux = []
                min_x, min_y = min(x_coords), min(y_coords)
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - (min_x / W))
                    data_aux.append(landmark.y - (min_y / H))
                
                if len(data_aux) == 42:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict.get(int(prediction[0]), "?")
                    
                    if predicted_character == self.last_predicted_character:
                        if self.character_start_time is None:
                            self.character_start_time = time.time()
                        elif time.time() - self.character_start_time >= 1.5:
                            self.recognized_sentence += predicted_character
                            self.root.after(0, lambda: self.label_sentence.config(
                                text="Recognized: " + self.recognized_sentence))
                            self.character_start_time = None
                    else:
                        self.last_predicted_character = predicted_character
                        self.character_start_time = time.time()
                
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)
            
            final_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(final_img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.root.after(0, lambda: self.update_image(imgtk))

    def update_image(self, imgtk):
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

    def add_space(self):
        self.recognized_sentence += " "
        self.label_sentence.config(text="Recognized: " + self.recognized_sentence)

    def clear_text(self, clear_all):
        if clear_all:
            self.recognized_sentence = ""
        else:
            self.recognized_sentence = self.recognized_sentence[:-1] if self.recognized_sentence else ""
        self.label_sentence.config(text="Recognized: " + self.recognized_sentence)

    def speak_text(self):
        if not self.recognized_sentence:
            return
            
        try:
           
            self.engine.say(self.recognized_sentence)
            self.engine.runAndWait()
            
            
            if not self.offline_mode.get():
                try:
                    from gtts import gTTS
                    import pygame
                    
                    tts = gTTS(text=self.recognized_sentence, lang='en')
                    output_file = "output.mp3"
                    
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    
                    tts.save(output_file)
                    pygame.mixer.init()
                    pygame.mixer.music.load(output_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    pygame.mixer.quit()
                    os.remove(output_file)
                except Exception:
                   
                    pass
        except Exception as e:
            self.label_translated.config(text=f"Error with speech: {str(e)}")

    def offline_translate(self, text, target_lang_name):
        words = text.lower().split()
        translated_words = []
        
        for word in words:
            if word == "iloveyou":
                word = "i love you"
                
            if word in offline_translations["English"] and target_lang_name in offline_translations["English"][word]:
                translated_words.append(offline_translations["English"][word][target_lang_name])
            else:
                translated_words.append(word)
        
        return " ".join(translated_words)

    def translate_text(self):
        if not self.recognized_sentence:
            messagebox.showinfo("Translation", "No text to translate")
            return

        try:
            selected_lang = self.selected_language.get()
            target_lang_code = self.language_options[selected_lang]
            
            
            if self.offline_mode.get():
                # Use offline translation
                translated_text = self.offline_translate(self.recognized_sentence, selected_lang)
                limited_translation = True
            else:
                
                try:
                    
                    from gtts import gTTS
                    
                   
                    test_tts = gTTS(text="test", lang="en")
                    
                    
                    try:
                        import requests
                        
                       
                        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target_lang_code}&dt=t&q={self.recognized_sentence}"
                        response = requests.get(url)
                        
                        if response.status_code == 200:
                            
                            data = response.json()
                            translated_text = ''.join([sentence[0] for sentence in data[0]])
                            limited_translation = False
                        else:
                            
                            translated_text = self.offline_translate(self.recognized_sentence, selected_lang)
                            limited_translation = True
                    except Exception:
                       
                        translated_text = self.offline_translate(self.recognized_sentence, selected_lang)
                        limited_translation = True
                except Exception:
                    
                    translated_text = self.offline_translate(self.recognized_sentence, selected_lang)
                    limited_translation = True

           
            self.label_translated.config(text=f"Translated: {translated_text}")

          
            if limited_translation:
                messagebox.showinfo("Translation Info", 
                                  "Using basic translation dictionary.\n"
                                  "Only common phrases are fully supported.")

            
            choice = messagebox.askyesno("Translation", 
                                      f"Translated text:\n{translated_text}\n\nWould you like to hear the translation?")
            if choice:
                self.speak_translated_text(translated_text, target_lang_code)

        except Exception as e:
            messagebox.showerror("Translation Error", str(e))
            
            try:
                translated_text = self.offline_translate(self.recognized_sentence, selected_lang)
                self.label_translated.config(text=f"Translated: {translated_text}")
            except:
                self.label_translated.config(text="Translation failed. Please try again.")

    def speak_translated_text(self, text, target_lang):
        try:
            
            self.engine.say(text)
            self.engine.runAndWait()
            
            
            if not self.offline_mode.get():
                try:
                    from gtts import gTTS
                    import pygame
                    
                    tts = gTTS(text=text, lang=target_lang)
                    output_file = "translated_output.mp3"
                    
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    
                    tts.save(output_file)
                    pygame.mixer.init()
                    pygame.mixer.music.load(output_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    pygame.mixer.quit()
                    os.remove(output_file)
                except Exception:
                    
                    messagebox.showinfo("TTS", "Using offline text-to-speech. Voice will be in English.")
            else:
                messagebox.showinfo("Offline TTS", "Using offline text-to-speech. Voice will be in English.")
        except Exception as e:
            messagebox.showerror("Speech Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()