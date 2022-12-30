import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
import sys
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os
from wav2spec import wav2spec
from AlexNetSpec import AlexNetSpec as ANS 

name = os.path.normpath(sys.argv[1])
print(name)
outputFolder = os.path.normpath(sys.argv[2])

wav2spec(outputFolder, name)
print("done with spectrogram")
arr = name.split('.')
arr = arr[0].split('\\')
filename = arr[len(arr)-1]
model = keras.models.load_model("model")

img = tf.keras.utils.load_img(
    outputFolder + '\\' + filename + ".png", target_size=(ANS.HEIGHT, ANS.WIDTH)
)

print("done loading")
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format([int(x) for x in range(10)][np.argmax(score)], 100 * np.max(score))
)