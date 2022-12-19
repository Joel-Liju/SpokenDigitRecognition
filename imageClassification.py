import glob
import tensorflow as tf

batch_size = 32
img_height = 369
img_width = 496

train_ds = tf.keras.utils.image_dataset_from_directory(
  "/dataset",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  "/dataset",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

print(train_ds.class_names)