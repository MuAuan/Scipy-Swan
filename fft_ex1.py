from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 100, 1000, endpoint=False)
sig=[]
for t1 in t:
    if t1<20:
        sig1 = np.cos(2 * np.pi * 0.5 * t1)
        print("5",t1)
    elif t1<40:
        sig1 = np.cos(2 * np.pi * 1 * t1)
        print("10",t1)
    elif t1<60:
        sig1 = np.cos(2 * np.pi * 1.5 * t1)
        print("20",t1)
    elif t1<80:
        sig1 = np.cos(2 * np.pi * 2 * t1)
        print("30",t1)
    else:
        sig1 = np.cos(2 * np.pi * 2.5 * t1)
        print(t1)
    sig.append(sig1)
        
plt.plot(t, sig)
plt.axis([0, 100, -2, 2])
plt.show()

freq =fft(sig,1024)
Pyy = np.sqrt(freq*freq.conj())/1025 #np.abs(freq)/1025    #freq*freq.conj(freq)/1025
f = np.arange(1024)
plt.plot(f,Pyy)
plt.axis([0, 512, 0,0.2])
plt.show()

t1=np.linspace(0, 100, 1024, endpoint=False)
sig1=ifft(freq)
plt.plot(t1, sig1)
plt.axis([0, 100, -2, 2])
plt.show()