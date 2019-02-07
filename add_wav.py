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
    #ax1.set_ylim(-0.0075,0.0075)
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
    #ax3.set_ylim(0,0.000075)
    ax3.set_xlim(20,20000)
    ax3.set_xlabel('Freq[Hz]')
    ax3.set_ylabel('Power')
    ax3.set_xscale('log')
    ax3.plot(f*fr/44100,Pyy)
    
    plt.savefig(filename+'.jpg')
    plt.show()
    
    plt.close()  
    ws.close()

def fileOpen(filename):
    wf = wave.open(filename+".wav", "r")
    
    printWaveInfo(wf)
    fr = wf.getframerate()
    fn = wf.getnframes()
    fs = float(fn / fr)
    
    # ストリームを開く
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # チャンク単位でストリームに出力し音声を再生
    CHANNELS = wf.getnchannels()
    width = wf.getsampwidth()
    return wf,stream,fr,fn,fs,width,CHANNELS
    
if __name__ == '__main__':
    filename1=input('input original filename=')
    wf1,stream1,fr1,fn1,fs1,width1,CHANNELS1 = fileOpen(filename1)

    filename2=input('input original filename=')
    wf2,stream2,fr2,fn2,fs2,width2,CHANNELS2 = fileOpen(filename2)
    
    fc1=int(input('input factor1='))
    fc2=int(input('input factor2='))
    
    frames = []
    for i in range(0, int(fr1 / 1024 *fs1+0.5)):
        data = wf1.readframes(1024)
        g1 = fc1*np.frombuffer(data, dtype= "int16") #/32768.0    # -1～1に正規化 #g1は演算できる
        frames.append(g1)
        stream1.write(g1)
        
    for i in range(0, int(fr2 / 1024 *fs2+0.5)):
        data2 = wf2.readframes(1024)
        g2 = fc2*np.frombuffer(data2, dtype= "int16") #/32768.0    # -1～1に正規化　#g2は演算できる
        frames.append(g2)
        stream2.write(g2)

    loff1 = wf1.getnframes()/1024 #215 #len(frames)
    loff2 = wf2.getnframes()/1024 #215 #len(frames)

    wr = wave.open('wav/'+filename1+filename2+'_out.wav', 'wb')
    wr.setnchannels(CHANNELS2)
    wr.setsampwidth(width2)  #width=2 ; 16bit
    wr.setframerate(fr2)
    s=int((loff1+loff2)*(fs1+fs2)/(fs1+fs2))
    wr.writeframes(b''.join(frames[0:s])) 
    #wr.close()

    fn = wr.getnframes()
    fs = float(fn / wr.getframerate())
    print(fn,fs)
    plot_wav('wav/'+filename1+filename2+'_out',0,fs)
        
    t1=float(input('input t1='))
    t2=float(input('input t2='))
    t3=float(input('input t3='))
    t4=float(input('input t4='))

    wr = wave.open('wav/'+filename1+filename2+'_out2.wav', 'wb')
    wr.setnchannels(CHANNELS2)
    wr.setsampwidth(width2)  #width=2 ; 16bit
    wr.setframerate(fr2)
    s1=int((loff1+loff2)*(t1)/(fs1+fs2))
    s2=int((loff1+loff2)*(t2)/(fs1+fs2))
    s3=int((loff1+loff2)*(t3)/(fs1+fs2))
    s4=int((loff1+loff2)*(t4)/(fs1+fs2))
    print(t1,t2,t3,t4,s1,s2,s3,s4)
    wr.writeframes(b''.join(frames[s1:s2])) 
    wr.writeframes(b''.join(frames[s3:s4])) 
    #wr.close()

    fn = wr.getnframes()
    fs = float(fn / wr.getframerate())
    print(fn,fs)
    plot_wav('wav/'+filename1+filename2+'_out2',0,fs)
    
    stream1.close()
    stream2.close()
        
    exit()
