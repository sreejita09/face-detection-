import os
import cv2
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def preprocess_data(raw_data_folder, target_size=(224, 224)):
    images = load_images_from_folder(raw_data_folder)
    processed_images = [preprocess_image(img, target_size) for img in images]
    return np.array(processed_images)