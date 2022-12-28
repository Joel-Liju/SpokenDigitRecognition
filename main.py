import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
import sys
import pyaudioTest
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from tkinter import ttk
from tkinter import *
from tkinter import filedialog as fd
from PIL import ImageTk, Image
import os

data = []

model = keras.models.load_model("model")

# create the root window
window = Tk()
window.title('Digit Recognition AI')
window.resizable(True, True)
window.geometry('500x400')


samplerate = IntVar()
samplerate.set(44100)

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
        runModel["state"] = "enabled"
        global data
        sampleratelocal,data = wavfile.read(filename)
        samplerateBox.delete(0,100)
        # samplerateBox.insert(0,sampleratelocal)
        samplerate.set(sampleratelocal)
        selectAudio["state"] = "disabled"
        playButton["state"] = "enabled"
    else: 
        runModel["state"] = "disabled"

def run():
    global samplerate,data
    global directory
    global model
    global filename
    print(samplerate.get())
    if filename == "":
        filename = "dummy"
    # samplerate, data = wavfile.read(filename)
    f, t, Sxx = signal.spectrogram(data, samplerate.get())

    img_height = 128
    img_width = 192

    # Set the size of the image
    figure = plt.figure()
    figure.set_size_inches(img_width/figure.get_dpi(), img_height/figure.get_dpi()) # convert pixels to inches

    # remove the axis and set the margins
    axes = plt.Axes(figure, [0., 0., 1., 1.])
    axes.set_axis_off()
    figure.add_axes(axes)
    axes.pcolormesh(t, f, Sxx, shading='gouraud')
    axes.xaxis.set_major_locator(plt.NullLocator())
    axes.yaxis.set_major_locator(plt.NullLocator())

    #Saving the spectrogram as a png
    arr = filename.split('.')
    #directory,name = arr[0].split('/')
    arr = arr[0].split('/')
    name = arr[len(arr)-1]
    imgName = "testdata/" + name + ".png"
    figure.savefig(imgName, bbox_inches="tight", pad_inches = 0)
    plt.close(figure)
    print("done with spectrogram")
    
    img = tf.keras.utils.load_img(
        imgName, target_size=(img_height, img_width)
    )
    lblImg = ImageTk.PhotoImage(Image.open(imgName))
    d.configure(image=lblImg)
    d.image = lblImg

    print("done loading")
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    c.config(text="This image most likely belongs to {} with a {:.2f} percent confidence."
                    .format([int(x) for x in range(10)][np.argmax(score)], 100 * np.max(score)))
    return
a = Label(window ,text = "File: ")
a.grid(row = 0, column = 0)
b = Label(window ,text = "")
b.grid(row = 0,column = 1)

selectAudio = ttk.Button(
    window,
    text='Select Audio',
    command=select_file
    )
selectAudio.grid(row=1, column=0)

runModel = ttk.Button(
    window,
    text='Run',
    command=run,
    state= "disabled"
    )
runModel.grid(row=1, column=1)

def recorder():
    global data
    data = pyaudioTest.record(fs=samplerate.get())
    playButton["state"] = "enabled"
    clearButton["state"] = "enabled"
    recordButton["state"] = "disabled"
    selectAudio["state"] = "disabled"
    runModel["state"] = "enabled"
recordButton = ttk.Button(
    window,
    text="Record",
    command= recorder
)
recordButton.grid(row=1,column=2)

def play():
    global data
    pyaudioTest.sound(data,fs=samplerate.get())
playButton = ttk.Button(
    window,
    text="Play recorded",
    command= play,
    state="disabled"
)
playButton.grid(row=1,column=3)


def clearer():
    global data
    data = []
    clearButton["state"] = "disabled"
    recordButton["state"] = "enabled"
    playButton["state"] = "disabled"
    selectAudio["state"] = "enabled"
    
clearButton = ttk.Button(
    window,
    text="Clear",
    command= clearer,
    state="disabled"
)
clearButton.grid(row=1,column=4)
    
samplerateBox = Entry(window)
# samplerateBox.pack()
samplerateBox.grid(row=1,column=7)
t = Label(window, textvariable=samplerate)
t.grid(row=2, column=7)


def updatesamplerate():
    samplerate.set(samplerateBox.get())

updatebutton = ttk.Button(
    window,
    text="Update Samplerate",
    command=updatesamplerate,
)
updatebutton.grid(row=2,column=6)

c = Label(window)
c.grid(row=2, columnspan=2)

d = Label(window)
d.grid(row=3)



window.mainloop()
