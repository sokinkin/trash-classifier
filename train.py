# train.py

import tensorflow as tf
from data_loader import load_datasets
from model import build_model
from utils import plot_history

def main():
    data_dir = "output_dataset"  # base directory of your split data
    img_height, img_width = 150, 150
    batch_size = 32
    num_classes = 6
    epochs = 50

    train_ds, val_ds, _ = load_datasets(data_dir, img_height, img_width, batch_size)

    model = build_model(input_shape=(img_height, img_width, 3), num_classes=num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor='val_loss', save_best_only=True)
    ]
    
    # Train the model
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    plot_history(history)
    model.save("final_model.h5")

if __name__ == '__main__':
    main()
