# data_loader.py

import os
import tensorflow as tf

def load_datasets(data_dir, img_height=150, img_width=150, batch_size=32):
    """
    Loads training, validation, and test datasets from the given directory.
    
    Assumes that `data_dir` contains three subdirectories: `train`, `val`, and `test`,
    each with subfolders for each class.
    
    Parameters:
      data_dir (str): Path to the base directory containing 'train', 'val', 'test'
      img_height (int): Height to which images will be resized.
      img_width (int): Width to which images will be resized.
      batch_size (int): Batch size for the dataset.
      
    Returns:
      train_ds, val_ds, test_ds: The training, validation, and test datasets.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )
    
    # Normalize pixel values to [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds  = test_ds.map(lambda x, y: (normalization_layer(x), y))
    
    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    data_dir = "output_dataset"
    train_ds, val_ds, test_ds = load_datasets(data_dir)
    print("Datasets loaded successfully.")
