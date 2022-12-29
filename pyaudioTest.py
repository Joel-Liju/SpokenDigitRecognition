"""PyAudio Example: Play a WAVE file."""

import wave
import sys
import time
import numpy as np
from scipy.io import wavfile
import pyaudio
import collections

BUFFER_SIZE = 1024
p = pyaudio.PyAudio()

def record(recording, fs=44100):
    # Open new audio input stream
    global stream
    global buffer
    buffer = []
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=BUFFER_SIZE)
    #stream.start_stream()
    # Continuously record audio from the stream and save it to the buffer
    while True:
        # Read from the stream
        data = stream.read(BUFFER_SIZE)
        # Convert the data to a NumPy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        # Add the audio data to the buffer
        buffer.extend(audio_data)
        if len(buffer) >= (fs*3) or not recording:
            stream.stop_stream()
            stream.close()
            p.terminate()
            break
        

def stop_record():
    # Save data
    array = np.asarray(buffer, dtype='int16')
    return array

# This method plays back the recoded auio to the user
def sound(array, fs=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=len(array.shape), rate=fs, output=True)
    stream.write(array.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

