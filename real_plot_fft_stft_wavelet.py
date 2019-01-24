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
CHANNELS = 1             # 1;monoral 2;ステレオ
RATE = 22100             # 22.1kHz 44.1kHz
RECORD_SECONDS = 5       # 5秒録音
WAVE_OUTPUT_FILENAME = "output2.wav"

s=1
while True:
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    print("* recording")
    
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

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
    amp = max(data)

    plt.figure(figsize=(12, 10))
    # ステレオ前提 > monoral
    sig = np.frombuffer(data, dtype="int16")  /32768.0
    t = np.linspace(0,fs, fn/2, endpoint=False)
    plt.subplot(311)
    plt.xlim([0, 5])
    plt.plot(t, sig)
    plt.pause(1)
    #plt.close()

    nperseg = 256
    #sig = np.frombuffer(data, dtype="int16")/32768.0
    #print('fs',fs, fn)
    f, t, Zxx = signal.stft(sig, fs=fs*fn/50, nperseg=nperseg)
    plt.subplot(312)
    plt.pcolormesh(t, 5*f, np.abs(Zxx), cmap='hsv')
    plt.ylim([10*f[1], 10*f[-1]])
    plt.xlim([0, 5])
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.yscale('log')
    plt.pause(1)
    #plt.close()

   
    freq =fft(sig,int(fn/2))
    Pyy = np.sqrt(freq*freq.conj())*2/fn #np.abs(freq)/1025    #freq*freq.conj(freq)/1025
    f = np.arange(int(fn/2))
    plt.subplot(313)
    plt.plot(f,Pyy)
    plt.axis([100, max(f)/2, 0,0.00005])  #max(Pyy)])
    plt.xscale('log')
    plt.pause(1)
    #plt.close()
    plt.savefig('figure'+str(s)+'.png')
    s += 1
       
    Fs = 1/0.0002
    omega0 = 5 #0.2 #1 #2 #8
    # (1)　Freqを指定してcwt
    x = np.linspace(0,fs, fn/2, endpoint=False)
    freqs=np.arange(10,2000,2.5)
    r=pycwt.cwt_f(sig,freqs,Fs,pycwt.Morlet(omega0))
    rr = np.abs(r)
    #fig=plt.subplot(413)
    plt.rcParams['figure.figsize'] = (10, 6)
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])
    ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax1)
    ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])

    ax1.plot(x, sig, 'k')

    img = ax2.imshow(np.flipud(rr), extent=[0, 5,100, 20000], aspect='auto', cmap='hsv') 
    twin_ax = ax2
    twin_ax.set_yscale('log')
    twin_ax.set_xlim(0, 5)
    twin_ax.set_ylim(100, 20000)
    ax2.tick_params(which='both', labelleft=False, left=False)
    twin_ax.tick_params(which='both', labelleft=True, left=True, labelright=False)
    fig.colorbar(img, cax=ax3)
    plt.pause(1)
    plt.savefig('figure_'+str(s)+'.png')
    s += 1
    