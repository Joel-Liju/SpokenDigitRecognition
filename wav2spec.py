# Alexander Freer
# 6452551
# This is to convert a .wav into spectrogram
from scipy.io import wavfile
from scipy import signal
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import sys
from AlexNetSpec import AlexNetSpec as ANS 
import os


def wav2spec(outputFolder, name, samplerate=0, data=[], loadWav=True):
    if loadWav :
        samplerate, data = wavfile.read(name)

    f, t, Sxx = signal.spectrogram(data, samplerate, mode = "magnitude")

    # Set the size of the image
    figure = plt.figure()
    figure.set_size_inches(ANS.WIDTH/figure.get_dpi(), ANS.HEIGHT/figure.get_dpi()) # convert pixels to inches

    # remove the axis and set the margins
    axes = plt.Axes(figure, [0., 0., 1., 1.])
    axes.set_axis_off()
    figure.add_axes(axes)
    axes.pcolormesh(t, f, Sxx, shading='gouraud')
    axes.xaxis.set_major_locator(plt.NullLocator())
    axes.yaxis.set_major_locator(plt.NullLocator())

    #Saving the spectrogram as a png
    arr = name.split('.')
    arr2 = arr[0].split('\\')
    filename = arr2[len(arr2)-1]
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    figure.savefig(outputFolder + '\\' + filename + ".png", bbox_inches="tight", pad_inches = 0)
    plt.close(figure)
   
def main():
    dataFolder = str(sys.argv[1])
    outputFolder = str(sys.argv[2])
    fNames = glob.glob(dataFolder+"/**/*.wav", recursive=True)
    for name in tqdm(fNames) :
        wav2spec(outputFolder, name)

if __name__ == "__main__":
    main()