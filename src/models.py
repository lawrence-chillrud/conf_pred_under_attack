# File: models.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 03/11/2023
# Description: Models defined here
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Sequential

def init_model(input_shape=(224, 224, 3), n_classes=100, freeze_base=True):
    # Load EfficientNetV2B0 with pre-trained ImageNet weights
    en_base = EfficientNetV2B0(
        weights='imagenet', 
        input_shape=input_shape, 
        classes=n_classes, 
        include_top=False
    )
    
    # Freeze EfficientNetV2B0 layers..!
    if freeze_base: 
        for layer in en_base.layers: layer.trainable = False
    
    model = Sequential([
        en_base,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])

    return model