import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# Load Dataset
def load_images(data_path, mask_path, img_size=(256, 256)):
    images, masks = [], []
    for file in os.listdir(data_path):
        img = cv2.imread(os.path.join(data_path, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size) / 255.0
        images.append(img)

        mask_file = os.path.join(mask_path, file)
        if os.path.exists(mask_file):
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size) / 255.0
            masks.append(mask)

    return np.array(images).reshape(-1, *img_size, 1), np.array(masks).reshape(-1, *img_size, 1)

# Define U-Net Model
def unet_model(input_shape=(256, 256, 1)):
    inputs = keras.Input(shape=input_shape)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    u1 = layers.UpSampling2D((2, 2))(c3)
    u1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    m1 = layers.Concatenate()([u1, c2])

    u2 = layers.UpSampling2D((2, 2))(m1)
    u2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    m2 = layers.Concatenate()([u2, c1])

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(m2)

    model = keras.Model(inputs, outputs)
    return model

# Train Model
def train_model():
    data_path = "path_to_images"
    mask_path = "path_to_masks"
    images, masks = load_images(data_path, mask_path)
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

    model = unet_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=8)
    model.save("unet_model.h5")

train_model()

