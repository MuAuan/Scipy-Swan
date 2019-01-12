from swan import pycwt
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 20, 0.01)
#y = np.sin(2 * np.pi * 2 * x) * 2 + np.sin(2 * np.pi * 5 * x) * 2  + np.sin(2 * np.pi * 10 * x)   
amp=1
y=[]
for t1 in x:
    if t1<3:
        sig1 = amp*np.cos(2 * np.pi * 0.5 * t1)  #+np.cos(2 * np.pi * 8 * t)  + np.cos(2 * np.pi * 15 * t)    #*np.exp(-0.1*t) *5
        
    elif t1<8:
        sig1 = amp*np.cos(2 * np.pi * 4 * t1)
        
    elif t1<12:
        sig1 = amp*np.cos(2 * np.pi * 7 * t1)
        
    elif t1<15:
        sig1 = amp*np.cos(2 * np.pi * 12 * t1)
        
    else:
        sig1 = amp*np.cos(2 * np.pi * 15 * t1)
        
    y.append(sig1)

plt.plot(x, y)
plt.show()

Fs = 1/0.01
omega0 = 0.2 #1 #2 #8
# (1)　Freqを指定してcwt
freqs=np.arange(0.1,20,0.025)
r=pycwt.cwt_f(y,freqs,Fs,pycwt.Morlet(omega0))
rr=np.abs(r)

plt.rcParams['figure.figsize'] = (10, 6)
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])
ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax1)
ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])

ax1.plot(x, y, 'k')

img = ax2.imshow(np.flipud(rr), extent=[0, 20,0.1, 20], aspect='auto')  #, interpolation='nearest')
twin_ax = ax2
twin_ax.set_yscale('log')
twin_ax.set_xlim(0, 20)
twin_ax.set_ylim(0.1, 20)
ax2.tick_params(which='both', labelleft=False, left=False)
twin_ax.tick_params(which='both', labelleft=True, left=True, labelright=False)
fig.colorbar(img, cax=ax3)
plt.show()

plt.plot(freqs,rr)
plt.xscale('log')
plt.show()