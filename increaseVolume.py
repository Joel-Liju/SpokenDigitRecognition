import glob
import sys
from scipy.io import wavfile
from tqdm import tqdm
import numpy as np

dataFolder = str(sys.argv[1])

file_names = glob.glob(dataFolder+"/**/*.wav", recursive=True)

for name in tqdm(file_names):
    sample_rate, data = wavfile.read(name)

    # Determine the maximum and minimum values of the audio data
    data_max = data.max()
    data_min = data.min()

    # Calculate the scaling factor to prevent clipping
    scaling_factor = max((32767/data_max), abs(32767/data_min))

    # Max volume
    amp = 0.6

    # Increase the volume to amp
    data = data * scaling_factor * amp

    data = data.astype(np.int16)

    # Save the modified audio data to a new file
    wavfile.write(name, sample_rate, data)