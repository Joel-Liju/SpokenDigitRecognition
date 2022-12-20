import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
import sys
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os

name = os.path.normpath(sys.argv[1])
print(name)
outputFolder = os.path.normpath(sys.argv[2])

samplerate, data = wavfile.read(name)

f, t, Sxx = signal.spectrogram(data, samplerate)
#plt.pcolormesh(t, f, Sxx, shading='gouraud')

# Set the size of the image
figure = plt.figure()
figure.set_size_inches(192/figure.get_dpi(), 128/figure.get_dpi()) # convert pixels to inches

# remove the axis and set the margins
axes = plt.Axes(figure, [0., 0., 1., 1.])
axes.set_axis_off()
figure.add_axes(axes)
axes.pcolormesh(t, f, Sxx, shading='gouraud')
axes.xaxis.set_major_locator(plt.NullLocator())
axes.yaxis.set_major_locator(plt.NullLocator())

#Saving the spectrogram as a png
arr = name.split('.')
print(arr)
directory,filename = arr[0].split('\\')
figure.savefig(outputFolder + '\\' + filename + ".png", bbox_inches="tight", pad_inches = 0)
plt.close(figure)
print("done with spectrogram")

model = keras.models.load_model("model")

img_height = 128
img_width = 192
img = tf.keras.utils.load_img(
    outputFolder + '\\' + filename + ".png", target_size=(img_height, img_width)
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