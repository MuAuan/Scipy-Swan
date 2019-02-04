#coding: utf-8
import wave
import pyaudio
import matplotlib.pyplot as plt
import numpy as np

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
    origin = ws.readframes(ws.getnframes())
    data = origin[:fn]
    sig = np.frombuffer(data, dtype="int16")  /32768.0
    t = np.linspace(0,fs, fn/2, endpoint=False)
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(211)
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Signal')

    ax1.set_ylim(-0.0075,0.0075)
    ax1.set_xlim(t1,t2)
    ax1.plot(t, sig)
    
    plt.show()  
    #ws.close()
    
    
if __name__ == '__main__':
    filename=input('input original filename=')
    wf = wave.open(filename+".wav", "r")
    
    printWaveInfo(wf)
    fs = float(wf.getnframes()) / wf.getframerate()
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
    #print(fs*RATE)
    """
    chunk = int(fs*RATE) #220160 #1024
    data = wf.readframes(chunk)
    stream.write(data)
    """
    frames = []
    for i in range(0, int(RATE / 1024 * int(fs+0.5)+0.5)):
        data = wf.readframes(1024)
        frames.append(data)
        stream.write(data)
    print(int(RATE / 1024 * int(fs+0.5)+0.5))
    
    t1=float(input('input t1='))
    t2=float(input('input t2='))
    plot_wav(filename,t1,t2)
    
    loff = wf.getnframes()/1024 #215 #len(frames)
    print(fs,loff,loff*t1/fs,loff*t2/fs)
    #wf.close()
    
    wr = wave.open(filename+'_out.wav', 'wb')
    wr.setnchannels(CHANNELS)
    wr.setsampwidth(width)  #width=2 ; 16bit
    wr.setframerate(RATE)
    s1=int(loff*0.5*(t1)/fs)
    s2=int(loff*(t2)/fs)-s1
    print(fs,loff,s1,s2,t1,t2)
    wr.writeframes(b''.join(frames[s1:s2]))  #int(loff*t2/fs)
    #wr.close()

    fn = int(wf.getnframes()*(s2-s1)/loff+0.5)   #wr.getnframes()
    fs = float(fn / wr.getframerate())
    print(fn,fs)
    plot_wav(filename+'_out',0,fs)
    
    stream.close()
    p.terminate()
    
    exit()
