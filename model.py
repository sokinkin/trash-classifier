# model.py

import tensorflow as tf
from tensorflow.keras import layers, models, initializers

def build_model(input_shape=(150, 150, 3), num_classes=6):
    """
    Builds and returns a CNN model using He Normal initialization.
    
    Parameters:
      input_shape (tuple): Shape of the input images.
      num_classes (int): Number of output classes.
      
    Returns:
      model: A tf.keras Model instance.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                      kernel_initializer=initializers.HeNormal(),
                      input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu',
                      kernel_initializer=initializers.HeNormal()),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu',
                      kernel_initializer=initializers.HeNormal()),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu',
                     kernel_initializer=initializers.HeNormal()),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
