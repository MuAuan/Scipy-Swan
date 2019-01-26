# -*- coding:utf-8 -*-

import pyaudio
import time
import matplotlib.pyplot as plt
import numpy as np
import wave
from scipy.fftpack import fft, ifft
from scipy import signal

N=100
CHUNK=1024*N
RATE=22100
CHANNELS = 1             # 1;monoral 2;ステレオ-
p=pyaudio.PyAudio()
WAVE_OUTPUT_FILENAME = "output.wav"
FORMAT = pyaudio.paInt16 # int16型

stream=p.open(	format = pyaudio.paInt16,
		channels = 1,
		rate = RATE,
		frames_per_buffer = CHUNK,
		input = True,
		output = True) # inputとoutputを同時にTrueにする

s=1
fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(311)
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Signal')
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax3.set_xlabel('Freq[Hz]')
ax3.set_xscale('log')
ax3.set_ylabel('Power')

while stream.is_active():

    fig.delaxes(ax1)
    fig.delaxes(ax3)
    ax1 = fig.add_subplot(311)
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Signal')
    ax3 = fig.add_subplot(313)
    
    input = stream.read(CHUNK)
    frames = []
    frames.append(input)
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
    #print("fn,fs",fn,fs)

    origin = wr.readframes(wr.getnframes())
    data = origin[:fn]
    wr.close()
    
    sig = np.frombuffer(data, dtype="int16")  /32768.0
    t = np.linspace(0,0.025*N, fn/2, endpoint=False)
    ax1.set_ylim(-0.0075,0.0075)
    ax1.set_xlim(0,0.025*N)
    ax1.plot(t, sig)
    
    nperseg = 1024
    f, t, Zxx = signal.stft(sig, fs=1024*50, nperseg=nperseg)
    ax2.pcolormesh(2.5*t, f, np.abs(Zxx), cmap='hsv')
    ax2.set_xlim(0,0.025*N)
    ax2.set_ylim(200,18000)
    ax2.set_yscale('log')
    
    freq =fft(sig,int(fn/2))
    Pyy = np.sqrt(freq*freq.conj())*2/fn
    f = np.arange(200,20000,(20000-200)/int(fn/2))
    ax3.set_ylim(0,0.000075)
    ax3.set_xlim(200,18000)
    ax3.set_xlabel('Freq[Hz]')
    ax3.set_ylabel('Power')
    ax3.set_xscale('log')
    ax3.plot(f,Pyy)
    
    plt.pause(0.01)
    plt.savefig('figure'+str(s)+'.png')
    s += 1
    output = stream.write(input)
	
stream.stop_stream()
stream.close()
p.terminate()

print( "Stop Streaming")