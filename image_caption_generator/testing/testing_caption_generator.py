# Importing Necessary Libraries
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception

# Extracting Features
def extract_features_pil(image, model):
    image = image.resize((299, 299))
    image = np.array(image)

    if image.shape[2] == 4:
        image = image[..., :3]

    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

# Mapping ID to Word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Caption Generator
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

# Deciding if OCR is needed
def needs_ocr(caption: str) -> bool:
    text_keywords = ["text", "label", "sign", "board", "writing", "word", "number", "letter"]
    return any(kw in caption.lower() for kw in text_keywords)

# Loading Models
max_length = 32
tokenizer = load(open("Image Caption Generator/tokenizer.p", "rb"))
model = load_model('Image Caption Generator/models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

#OpenCV Capture
cap = cv2.VideoCapture(0)  # webcam 0
print("Press 'c' to capture an image and generate caption. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Feed - Press 'c' to capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # Capture frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        photo = extract_features_pil(pil_img, xception_model)
        description = generate_desc(model, tokenizer, photo, max_length)
        description = description.replace("start", "").replace("end", "").strip()

        print("\nGenerated Caption:", description)
        print("OCR Trigger:", needs_ocr(description))

        plt.imshow(pil_img)
        plt.axis("off")
        plt.show()

    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()