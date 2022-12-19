# Alexander Freer
# 6452551
# This is to convert a .wav into spectrogram
from scipy.io import wavfile
from scipy import signal
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import sys

dataFolder = str(sys.argv[1])
fNames = glob.glob(dataFolder+"/*.wav")
for name in tqdm(fNames) :
    samplerate, data = wavfile.read(name)
    N = len(data)

    f, t, Sxx = signal.spectrogram(data, N)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    
    #remove the axis and set the margins
    plt.gca().set_axis_off()
    plt.margins(0,0)

    #Saving the spectrogram as a png
    arr = name.split('.')
    plt.savefig(arr[0] + ".png", bbox_inches="tight", pad_inches = 0)
