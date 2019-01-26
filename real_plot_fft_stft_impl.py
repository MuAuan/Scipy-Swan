import pyaudio
import wave
from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from swan import pycwt

CHUNK = 1024
FORMAT = pyaudio.paInt16 # int16型
CHANNELS = 1             # 1;monoral 2;ステレオ-
RATE = 22100             # 22.1kHz 44.1kHz
RECORD_SECONDS = 5       # 5秒録音
WAVE_OUTPUT_FILENAME = "output2.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK)
s=1

# figureの初期化
fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax2.axis([0, 5, 200,20000]) 
ax2.set_yscale('log')

while True:
    fig.delaxes(ax1)
    fig.delaxes(ax3)
    ax1 = fig.add_subplot(311)
    ax3 = fig.add_subplot(313)
    print("* recording")
    
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done recording")

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    wavfile = WAVE_OUTPUT_FILENAME
    wr = wave.open(wavfile, "rb")
    ch = CHANNELS #wr.getnchannels()
    width = p.get_sample_size(FORMAT) #wr.getsampwidth()
    fr = RATE  #wr.getframerate()
    fn = wr.getnframes()
    fs = fn / fr

    origin = wr.readframes(wr.getnframes())
    data = origin[:fn]
    wr.close()
    
    sig = np.frombuffer(data, dtype="int16")  /32768.0
    t = np.linspace(0,fs, fn/2, endpoint=False)
    ax1.axis([0, 5, -0.0075,0.0075]) 
    ax1.plot(t, sig)
    
    nperseg = 256
    f, t, Zxx = signal.stft(sig, fs=fs*fn/50, nperseg=nperseg)
    ax2.pcolormesh(t, 5*f, np.abs(Zxx), cmap='hsv')
    
    freq =fft(sig,int(fn/2))
    Pyy = np.sqrt(freq*freq.conj())*2/fn
    f = np.arange(int(fn/2))
    ax3.axis([200, 20000, 0,0.000075])
    ax3.set_xscale('log')
    ax3.plot(f,Pyy)
    plt.pause(1)
    
    plt.savefig('figure'+str(s)+'.png')
    s += 1
   
