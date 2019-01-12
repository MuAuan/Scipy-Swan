#scipy.signal.istft example
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.istft.html
#
import numpy as np  #added by author
from scipy import signal
import matplotlib.pyplot as plt

#Generate a test signal, a 2 Vrms sine wave at 50Hz corrupted by 0.001 V**2/Hz of white noise sampled at 1024 Hz.
#テスト信号、1024 Hzでサンプリングされた0.001 V ** 2 / Hzのホワイトノイズで破損した50 Hzの2 Vrmsの正弦波を生成します

fs = 1024
N = 10*fs
nperseg = 64 #2048 #1024 #128 #256 #512
amp = 2 * np.sqrt(2)
noise_power = 0.0001 * fs / 2
time = np.arange(N) / float(fs)
#carrier = amp * np.sin(2*np.pi*50*time)
#t = np.linspace(0, 100, 1000, endpoint=False)
carrier=[]
for t1 in time:
    if t1<1:
        sig1 = amp*np.cos(2 * np.pi * 5 * t1)  #+np.cos(2 * np.pi * 8 * t)  + np.cos(2 * np.pi * 15 * t)    #*np.exp(-0.1*t) *5
        print("5",t1)
    elif t1<2:
        sig1 = amp*np.cos(2 * np.pi * 10 * t1)
        print("10",t1)
    elif t1<4:
        sig1 = amp*np.cos(2 * np.pi * 20 * t1)
        print("20",t1)
    elif t1<6:
        sig1 = amp*np.cos(2 * np.pi * 50 * t1)
        print("30",t1)
    else:
        sig1 = amp*np.cos(2 * np.pi * 100 * t1)
        print(t1)
    carrier.append(sig1)
    
noise = np.random.normal(scale=np.sqrt(noise_power),
                         size=time.shape)
x = carrier + noise
plt.plot(time,x)
plt.show()

#Compute and plot the STFT’s magnitude.
#STFTの振幅を計算してプロットします

f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg)
plt.figure()
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
plt.ylim([f[1], f[-1]])
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale('log')
plt.show()

#Zero the components that are 10% or less of the carrier magnitude, then convert back to a time series via inverse STFT
#キャリア振幅の10％以下の成分をゼロにしてから、逆STFTを介して時系列に変換し直す

Zxx = np.where(np.abs(Zxx) >= amp/10, Zxx, 0)
_, xrec = signal.istft(Zxx, fs)

#Compare the cleaned signal with the original and true carrier signals.
#きれいにされた信号を元のそして本当の搬送波信号と比較

plt.figure()
plt.plot(time, x, time, xrec, time, carrier)
plt.xlim([5, 6])
plt.xlabel('Time [sec]')
plt.ylabel('Signal')
plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
plt.show()

#Note that the cleaned signal does not start as abruptly as the original, since some of the coefficients of the transient were also removed:
#トランジェントの係数の一部も削除されているため、クリーン化された信号は元の信号ほど急激には開始されません。
plt.figure()
plt.plot(time, x, time, xrec, time, carrier)
plt.xlim([0, 1])
plt.xlabel('Time [sec]')
plt.ylabel('Signal')
plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
plt.show()