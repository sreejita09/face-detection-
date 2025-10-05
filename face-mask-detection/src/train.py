import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model  # Assuming create_model is a function in model.py
from utils import save_model  # Assuming save_model is a function in utils.py

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'  # Assuming binary classification: mask/no mask
    )
    return train_generator

def train_model(train_data, epochs=10):
    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, epochs=epochs)
    return model

if __name__ == "__main__":
    data_directory = os.path.join('data', 'processed')  # Adjust path as necessary
    train_data = load_data(data_directory)
    model = train_model(train_data)
    save_model(model, 'face_mask_detection_model.h5')  # Save the model after training