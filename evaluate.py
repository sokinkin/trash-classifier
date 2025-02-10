# evaluate.py

import tensorflow as tf
from data_loader import load_datasets

def main():
    data_dir = "output_dataset"
    img_height, img_width = 150, 150
    batch_size = 32

    _, _, test_ds = load_datasets(data_dir, img_height, img_width, batch_size)

    model = tf.keras.models.load_model("best_model.keras")

    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2%}")

if __name__ == '__main__':
    main()
