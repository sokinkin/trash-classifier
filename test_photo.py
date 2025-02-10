import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

CLASS_LABELS = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def load_and_preprocess_image(img_path, img_height=150, img_width=150):
    """Loads and preprocesses an image for model prediction."""
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to [0,1]
    return img_array

def predict_image(model_path, img_path):
    """Loads a trained model and predicts the class of an input image."""
    model = tf.keras.models.load_model(model_path)

    img_array = load_and_preprocess_image(img_path)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    plt.imshow(image.load_img(img_path))
    plt.axis('off')
    plt.title(f"Predicted: {CLASS_LABELS[predicted_class]} ({confidence:.2%})")
    plt.show()

if __name__ == "__main__":
        model_path = "best_model.keras"
        img_path = "image.png"
        predict_image(model_path, img_path)
