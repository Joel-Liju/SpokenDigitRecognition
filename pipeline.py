import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
import sys

file = str(sys.argv[1])

model = keras.models.load_model("model")

img_height = 128
img_width = 192
img = tf.keras.utils.load_img(
    file, target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format([int(x) for x in range(10)][np.argmax(score)], 100 * np.max(score))
)
print(predictions)