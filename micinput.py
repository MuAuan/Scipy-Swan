# -*- coding: utf-8 -*-
#マイク0番からの入力を受ける。一定時間(RECROD_SECONDS)だけ録音し、ファイル名：入力名.wavで保存する。
 
import pyaudio
import sys
import time
import wave
 
if __name__ == '__main__':
    CHUNK = 1024 #1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  #monoral
    #サンプリングレート、マイク性能に依存
    RATE = 44100
    # 録音時間
    RECORD_SECONDS = input('Please input recoding time>>>')
    filename=input('input filename=')
    
    p = pyaudio.PyAudio()
    input_device_index = 0 #RasPi；mic 0番のとき、Windowsは不要だが、あっても動く
    
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * int(RECORD_SECONDS))):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename+'.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))  #width=2 ; 16bit
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()