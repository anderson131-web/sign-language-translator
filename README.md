# 🧠 Sign Language Translator 🤟

This desktop software uses real-time technology to detect hand signs which then converts into spoken words and translations across different  languages. The application combines Python programming with OpenCV along with fundamental machine learning model structures.



---

## 🔧 Features

-  Webcams enable immediate hand gesture detection functionality

-  The system transforms detected signs first into text and then into speech output

-  The system enables translation of detected words into several different languages

-  Lightweight ML-based gesture classifier

-  The system functions without internet connection and with internet connectivity

-  IP Camera reconfiguration support

---

##  How It Works

1. Use collect_imgs.py to gather training data for each gesture

2. Run create_dataset.py to process and label data

3. Train a model with train_classifier.py

4. Launch the translator app using inference_classifier.py

---

## Getting Started

```bash

git clone https://github.com/anderson131-web /sign-language-translator.git

cd sign-language-translator

pip install -r requirements.txt

python inference_classifier.py