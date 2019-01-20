from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt
import wave
import struct
from pylab import *
from scipy import signal

#wavfile = 'hirakegoma.wav'
wavfile = 'ohayo.wav'
wr = wave.open(wavfile, "rb")
ch = wr.getnchannels()
width = wr.getsampwidth()
fr = wr.getframerate()
fn = wr.getnframes()
fs = fn / fr

print('ch', ch)
print('frame', fn)
print('fr',fr)
print('sampling fs ', fs, 'sec')
print('width', width)
origin = wr.readframes(wr.getnframes())
data = origin[:fn]
wr.close()
amp = max(data)
print(amp)

print('len of origin', len(origin))
print('len of sampling: ', len(data))

# ステレオ前提 > monoral
sig = np.frombuffer(data, dtype="int16")  #/32768.0
t = np.linspace(0,fs, fn/2, endpoint=False)
plt.plot(t, sig)
plt.show()

freq =fft(sig,int(fn/2))
Pyy = np.sqrt(freq*freq.conj())*2/fn #np.abs(freq)/1025    #freq*freq.conj(freq)/1025
f = np.arange(int(fn/2))
plt.plot(f,Pyy)
plt.axis([10, max(f)/2, 0,max(Pyy)])
plt.xscale('log')
plt.show()

t1 = np.linspace(0,fs, fn/2, endpoint=False)
sig1=ifft(freq,fn/2)
plt.plot(t1, sig1)
plt.axis([0.1, fs, min(sig1), max(sig1)])
plt.show()