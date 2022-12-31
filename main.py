import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tkinter as tk
from scipy.io import wavfile
from scipy import signal
from tkinter import ttk
from tkinter import *
from tkinter import filedialog as fd
from PIL import ImageTk, Image
from AlexNetSpec import AlexNetSpec as ANS 
from wav2spec import wav2spec
import os
import threading
import pyaudio

data = []

model = keras.models.load_model("model") # this model is pulled from the model folder, which is the parameters for the neural network.

# create the root window
window = Tk()
window.title('Digit Recognition AI')
window.resizable(True, True)
window.geometry('700x400')

# changeable sample rate
samplerate = IntVar()
samplerate.set(8000)
# PyAudio configurations
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

filename = ""
directory = os.getcwd()

def select_file():
    """
    This function opens the file explorer, 
    and then you are able to select a wav file which contains the recording of the number being said.
    """
    global filename
    choosen = fd.askopenfilename(
        title='Select Audio',
        initialdir=directory,
        filetypes=[('WAV files', '*.wav')]) 
    if choosen != "" :
        filename = choosen
        filename = filename.replace('/', '\\')
        print(filename)
        arr = filename.split('\\')
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
    """
    this function takes the data which is the samples for the audio file, 
    then runs the machine learning model on the image, and then predicts what number it might be.
    """
    global samplerate,data
    global directory
    global model
    global filename
    

    if filename == "":
        filename = "dummy"
        wav2spec("testdata", filename, samplerate.get(), data, False)
    else: 
        wav2spec("testdata", filename)
        arr = filename.split('.')
        arr2 = arr[0].split('\\')
        filename = arr2[len(arr2)-1]
    
    print("done with spectrogram")
    imgName = "testdata\\" + filename + ".png"
    img = tf.keras.utils.load_img(
        imgName, target_size=(ANS.HEIGHT, ANS.WIDTH)
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
    """
    this function using a thread, starts recording the audio into a buffer.
    """
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
    """
    this function uses a buffer and puts the recorded audio into the buffer, which is then put into the variable data.
    This can be used by other functions to perform their tasks such as play function from the pyaudioTest. 
    """
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

# This method plays back the recoded auio to the user
def sound(array, fs=44100):
    """
    parameters:
        array -> this array contains the samples for the recorded audio using the buffer above.
        fs -> this is the sampling rate which is defaulted to 44100 samples per second.

    this function, takes the Pyaudio class, and then opens a stream and writes it onto the buffer, whcih then will be played 
    through the default Audio player.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=len(array.shape), rate=fs, output=True)
    stream.write(array.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

def play():
    """
    this function takes the data which contains the samples from the audio, and then plays it.
    """
    global data
    sound(data,fs=samplerate.get())

playButton = ttk.Button(
    window,
    text="Play recorded",
    command= play,
    state="disabled"
)
playButton.grid(row=1,column=4)

def clearer():
    """
    this function clears the audio data buffer.
    """
    global data,filename
    data = []
    filename = ""
    b.config(text="")
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