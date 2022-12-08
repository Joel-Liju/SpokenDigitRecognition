from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
import math

f = open('testdata/data.txt','r')
audio = [float(x) for x in f.readlines()]
f.close()

# Number of sample points
N = len(audio)
sample_rate = 44100.0
# sample spacing
T = 1.0 / sample_rate
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.array(audio) 
# print(y)
# print(y)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

from scipy import signal
from scipy.fft import fftshift
f, t, Sxx = signal.spectrogram(y, N)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()