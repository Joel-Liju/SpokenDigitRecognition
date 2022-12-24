import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
import sys
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from tkinter import ttk
from tkinter.ttk import Progressbar
from tkinter import *
from tkinter import filedialog as fd
import os

# create the root window
window = Tk()
window.title('Digit Recognition AI')
window.resizable(True, True)
window.geometry('300x400')

filename = ""
directory = os.getcwd()

def select_file():
    global filename
    choosen = fd.askopenfilename(
        title='Select Audio',
        initialdir=directory,
        filetypes=[('WAV files', '*.wav')]) 
    if choosen != "" :
        filename = choosen
        print(filename)
        arr = filename.split('/')
        name = arr[len(arr)-1]
        b.config(text=name)

    if filename != "": 
        btn1["state"] = "enable"
    else: btn1["state"] = "disabled"

def run():
    global filename
    pbar.start()
    samplerate, data = wavfile.read(filename)
    f, t, Sxx = signal.spectrogram(data, samplerate)

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
    arr = filename.split('.')
    print(arr)
    directory,name = arr[0].split('\\')
    figure.savefig(directory + '\\' + name + ".png", bbox_inches="tight", pad_inches = 0)
    plt.close(figure)
    print("done with spectrogram")

    model = keras.models.load_model("model")

    img_height = 128
    img_width = 192
    img = tf.keras.utils.load_img(
        directory + '\\' + name + ".png", target_size=(img_height, img_width)
    )

    print("done loading")
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    pbar.stop()
    return

a = Label(window ,text = "File: ")
a.grid(row = 0, column = 0)
b = Label(window ,text = "")
b.grid(row = 0,column = 1)

btn = ttk.Button(
    window,
    text='Select Audio',
    command=select_file
    )
btn.grid(row=1, column=0)

btn1 = ttk.Button(
    window,
    text='Run',
    command=run
    )
btn1.grid(row=1, column=1)
btn1["state"] = "disabled"

pbar = Progressbar(window, orient=HORIZONTAL, length=200, mode="indeterminate",takefocus=True)


# run the application
window.mainloop()
