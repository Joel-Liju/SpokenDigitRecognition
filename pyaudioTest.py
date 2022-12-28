"""PyAudio Example: Play a WAVE file."""

import wave
import sys
import time
import numpy as np
from scipy.io import wavfile
import pyaudio

p = pyaudio.PyAudio()

def record(fs=44100):
    # Open new audio input stream
    global stream
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True,
                    frames_per_buffer=1024)
    stream.start_stream()

def stop_record(fs=44100):
    # Stop the stream
    stream.stop_stream()

    # Determine the total time of recording 
    latency = stream.get_input_latency() 

    # Calculate the total number of frames in the stream
    num_frames = int(latency * fs)

    # Get all frames in one call
    buffer = stream.read(num_frames)

    # Save data and free resources 
    array = np.frombuffer(buffer, dtype='int16')
    stream.close()
    p.terminate()
    return array

# This method plays back the recoded auio to the user
def sound(array, fs=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=len(array.shape), rate=fs, output=True)
    stream.write(array.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

