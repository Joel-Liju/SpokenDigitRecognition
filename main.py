import numpy as np
import tensorflow as tf
from tensorflow import keras
import pyaudioTest
import matplotlib.pyplot as plt
import tkinter as tk
from scipy.io import wavfile
from scipy import signal
from tkinter import ttk
from tkinter import *
from tkinter import filedialog as fd
from PIL import ImageTk, Image
import os
import threading
import pyaudio

data = []

model = keras.models.load_model("model")

# create the root window
window = Tk()
window.title('Digit Recognition AI')
window.resizable(True, True)
window.geometry('700x400')

# changeable sample rate
samplerate = IntVar()
samplerate.set(44100)
# PyAudio configurations
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

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
        runButton["state"] = "enabled"
        global data
        sampleratelocal,data = wavfile.read(filename)
        samplerateBox.delete(0,100)
        samplerate.set(sampleratelocal)
        selectAudio["state"] = "disabled"
        playButton["state"] = "enabled"
        clearButton["state"] = "enabled"
        startRecording["state"] = "disabled"
    else: 
        runButton["state"] = "disabled"

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

runButton = ttk.Button(
    window,
    text='Run',
    command=run,
    state= "disabled"
    )
runButton.grid(row=1, column=1)



# Create a flag to control the recording loop
window.recording = False

def start_recording():
    # Enable the stop button
    stopRecording["state"] = "enabled"
    startRecording["state"] = "disabled"
    selectAudio["state"] = "disabled"
    runButton["state"] = "disabled"
    window.recording = True
    # Create a thread to run the recording loop
    recording_thread = threading.Thread(target=record_loop)
    recording_thread.start()

def record_loop():
    # Create a buffer to store the recorded audio
    buffer = bytes()
    
    # Create a PyAudio object
    p = pyaudio.PyAudio()

    # Open a stream to record audio
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=samplerate.get(),
                    input=True,
                    frames_per_buffer=CHUNK)

    # Start recording audio
    while True:
        buffer = buffer + stream.read(CHUNK)
        if not window.recording:
            break
    # Close the stream and PyAudio object
    stream.stop_stream()
    stream.close()
    p.terminate()
    global data
    data = np.frombuffer(buffer, dtype='int16')
    data = data[:-int(samplerate.get()*0.1)]


def stop_recording():
    # Set the flag to stop the recording loop
    window.recording = False
    # Disable the stop button and re-enable other buttons
    stopRecording["state"] = "disabled"
    clearButton["state"] = "enabled"
    playButton["state"] = "enabled"
    runButton["state"] = "enabled"
    
    

# Create the start button
startRecording = ttk.Button(text="Start recording", command=start_recording)
startRecording.grid(row=1,column=2)

# Create the stop button
stopRecording = ttk.Button(text="Stop recording", command=stop_recording)
stopRecording.grid(row=1,column=3)
stopRecording.config(state="disabled")

def play():
    global data
    pyaudioTest.sound(data,fs=samplerate.get())
playButton = ttk.Button(
    window,
    text="Play recorded",
    command= play,
    state="disabled"
)
playButton.grid(row=1,column=4)


def clearer():
    global data
    data = []
    clearButton["state"] = "disabled"
    startRecording["state"] = "enabled"
    playButton["state"] = "disabled"
    selectAudio["state"] = "enabled"
    runButton["state"] = "disabled"
    
clearButton = ttk.Button(
    window,
    text="Clear",
    command= clearer,
    state="disabled"
)
clearButton.grid(row=1,column=5)
    
samplerateBox = tk.Entry(window)

# Provide hint for user
samplerateBox.insert(0, "New sample rate")
samplerateBox.configure(fg="gray")

# Clear Entry when clicked
def on_focus(event):
    samplerateBox.delete(0, tk.END)
    samplerateBox.configure(fg="black")

samplerateBox.bind("<FocusIn>", on_focus)

samplerateBox.grid(row=1,column=8)
t = Label(window, textvariable=samplerate)
t.grid(row=2, column=8)

def updatesamplerate():
    playButton["state"] = "disabled"
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
