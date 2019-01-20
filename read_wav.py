import wave
from scipy import fromstring, int16
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#wavfile = 'hirakegoma.wav'
wavfile = 'ohayo.wav'
wr = wave.open(wavfile, "rb")
ch = wr.getnchannels()
width = wr.getsampwidth()
fr = wr.getframerate()
fn = wr.getnframes()

nperseg = 256 #4096 #2048 #1024 #256 #128 #32 #64 #512

print('ch', ch)
print('frame', fn)
fs = fn / fr
print('fr',fr)
print('sampling fs ', fs, 'sec')
print('width', width)
origin = wr.readframes(wr.getnframes())
data = origin[:fn]
wr.close()
amp = max(data)
print('amp',amp)

print('len of origin', len(origin))
print('len of sampling: ', len(data))

# ステレオ前提 > monoral
x = np.frombuffer(data, dtype="int16")  #/32768.0
print('max(x)',max(x))
t = np.linspace(0,fs, fn/2, endpoint=False)
plt.plot(t, x)
plt.show()

f, t, Zxx = signal.stft(x, fs=fs*fn/20, nperseg=nperseg)
plt.figure()
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
plt.ylim([f[1], f[-1]])
plt.xlim([0, 3])
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale('log')
plt.show()

#Zero the components that are 10% or less of the carrier magnitude, then convert back to a time series via inverse STFT
#キャリア振幅の10％以下の成分をゼロにしてから、逆STFTを介して時系列に変換し直す
maxZxx= max(data)
print('maxZxx',maxZxx)
Zxx = np.where(np.abs(Zxx) >= maxZxx*2, Zxx, 0)
plt.figure()
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
plt.ylim([f[1], f[-1]])
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale('log')
plt.show()
_, xrec = signal.istft(Zxx, fs)

#Compare the cleaned signal with the original and true carrier signals.
#きれいにされた信号を元のそして本当の搬送波信号と比較
t = np.linspace(0,fs, fn/2, endpoint=False)
plt.figure()
plt.plot(t, x, t, xrec)   #, time, carrier)
#plt.xlim([20, 75])
plt.xlabel('Time [sec]')
plt.ylabel('Signal')
#plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
plt.show()

plt.figure()
plt.plot(t, xrec-x)   #, time, carrier)
#plt.xlim([0, 0.1])
plt.xlabel('Time [sec]')
plt.ylabel('Signal')
#plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
plt.show()

# 書き出し
outf = './output/test.wav'
ww = wave.open(outf, 'w')
ww.setnchannels(ch)
ww.setsampwidth(width)
ww.setframerate(fr)
outd = x #xrec
print(len(x))
ww.writeframes(outd)
ww.close()

outf = './output/test1.wav'
ww = wave.open(outf, 'w')
ww.setnchannels(ch)
ww.setsampwidth(2*width)
ww.setframerate(2*fr)
maxrec=max(xrec)
outd = xrec/maxrec
print(max(outd),min(outd))
ww.writeframes(outd)
ww.close()