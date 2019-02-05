#coding: utf-8
import wave
import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import time
import wave
from scipy.fftpack import fft, ifft
from scipy import signal

def printWaveInfo(wf):
    """WAVEファイルの情報を取得"""
    print( "chn:", wf.getnchannels())
    print( "width:", wf.getsampwidth())
    print( "sampl rate:", wf.getframerate())
    print( "no. of frame:", wf.getnframes())
    print( "parames:", wf.getparams())
    print( "length(s):", float(wf.getnframes()) / wf.getframerate())

def plot_wav(filename,t1,t2):
    ws = wave.open(filename+'.wav', "rb")
    ch = ws.getnchannels()
    width = ws.getsampwidth()
    fr = ws.getframerate()
    fn = ws.getnframes()
    fs = fn / fr
    print(fn,fs)
    origin = ws.readframes(fn)
    data = origin[:]
    sig = np.frombuffer(data, dtype="int16")  /32768.0
    t = np.linspace(0,fs, fn, endpoint=False)
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Signal')
    ax1.set_ylim(-0.0075,0.0075)
    ax1.set_xlim(t1,t2)
    ax1.plot(t, sig)
    
    nperseg = 1024
    f, t, Zxx = signal.stft(sig, fs=fn, nperseg=nperseg)
    ax2.pcolormesh(fs*t, f/fs/2, np.abs(Zxx), cmap='hsv')
    ax2.set_xlim(t1,t2)
    ax2.set_ylim(20,20000)
    ax2.set_yscale('log')

    freq =fft(sig,int(fn))
    Pyy = np.sqrt(freq*freq.conj())*2/fn
    f = np.arange(20,20000,(20000-20)/int(fn)) #RATE11025,22050;N50,100
    ax3.set_ylim(0,0.000075)
    ax3.set_xlim(20,20000)
    ax3.set_xlabel('Freq[Hz]')
    ax3.set_ylabel('Power')
    ax3.set_xscale('log')
    ax3.plot(f*fr/44100,Pyy)
    
    plt.savefig(filename+'.jpg')
    plt.show()
    
    plt.close()  
    ws.close()
    
    
if __name__ == '__main__':
    filename=input('input original filename=')
    wf = wave.open(filename+".wav", "r")
    
    printWaveInfo(wf)
    fr = wf.getframerate()
    fn = wf.getnframes()
    fs = float(fn / fr)
    plot_wav(filename,0,fs)
    
    # ストリームを開く
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # チャンク単位でストリームに出力し音声を再生
    CHANNELS = wf.getnchannels()
    width = wf.getsampwidth()
    RATE = wf.getframerate()
    #fn = wf.readframes(wf.getnframes())
    fn = wf.getnframes()
    fs = float(wf.getnframes()) / wf.getframerate()

    frames = []
    for i in range(0, int(RATE / 1024 *fs+0.5)):
        data = wf.readframes(1024)
        frames.append(data)
        stream.write(data)
    print(int(RATE / 1024 * fs+0.5))
    
    t1=float(input('input t1='))
    t2=float(input('input t2='))
    plot_wav(filename,t1,t2)
        
    loff = wf.getnframes()/1024 #215 #len(frames)
    print(fs,loff,loff*t1/fs,loff*t2/fs)
    #wf.close()
    
    wr = wave.open('wav/'+filename+'_out.wav', 'wb')
    wr.setnchannels(CHANNELS)
    wr.setsampwidth(width)  #width=2 ; 16bit
    wr.setframerate(RATE)
    s1=int(loff*(t1)/fs)
    s2=int(loff*(t2)/fs)
    print(fs,loff,s1,s2,t1,t2)
    wr.writeframes(b''.join(frames[s1:s2]))  #int(loff*t2/fs)
    #wr.close()

    fn = wr.getnframes()
    fs = float(fn / wr.getframerate())
    print(fn,fs)
    plot_wav('wav/'+filename+'_out',0,fs)
    
    stream.close()
    p.terminate()
    
    exit()
