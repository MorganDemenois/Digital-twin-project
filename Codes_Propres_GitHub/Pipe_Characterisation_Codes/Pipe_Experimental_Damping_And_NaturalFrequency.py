import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def f(x,b,order):
    ans = 0
    for i in range(order+1):
        ans += b[i]*x**(order-i)
    return ans

z,x = np.loadtxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\XYZT_tip_0Hz1.txt")
x2 = x[2000:]
peaks2 = scipy.signal.find_peaks(x2,height=[0,0.5],distance=10)[0][:-1]
peaks2_n = scipy.signal.find_peaks(x2,height=[-0.5,0],distance=10)[0][:-1]
plt.plot(x2[peaks2])
plt.show()
b2 = np.polyfit(peaks2/150,np.log(x2[peaks2]),1)
plt.plot(peaks2/150,f(peaks2/150,b2,1))
plt.plot(peaks2/150,np.log(x2[peaks2]))
plt.show()
fft2 = np.fft.rfft(x2)
plt.plot(x2)
plt.plot(peaks2,x2[peaks2])
plt.plot(peaks2_n,x2[peaks2_n])
plt.show()
plt.plot((np.real(fft2[:200])**2+np.imag(fft2[:200])**2)**0.5)
plt.show()
f2 = 150*(peaks2.shape[0]-1)/(peaks2[-1]-peaks2[0])

z,x = np.loadtxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\XYZT_tip_0Hz2.txt")
x2 = x[2000:]
peaks2 = scipy.signal.find_peaks(x2,height=[0,0.5],distance=10)[0][:-1]
peaks2_n = scipy.signal.find_peaks(x2,height=[-0.5,0],distance=10)[0][:-1]
plt.plot(x2[peaks2])
plt.show()
b2 = np.polyfit(peaks2/150,np.log(x2[peaks2]),1)
plt.plot(peaks2/150,f(peaks2/150,b2,1))
plt.plot(peaks2/150,np.log(x2[peaks2]))
plt.show()
fft2 = np.fft.rfft(x2)
plt.plot(x2)
plt.plot(peaks2,x2[peaks2])
plt.plot(peaks2_n,x2[peaks2_n])
plt.show()
plt.plot((np.real(fft2[:200])**2+np.imag(fft2[:200])**2)**0.5)
plt.show()
f2 = 150*(peaks2.shape[0]-1)/(peaks2[-1]-peaks2[0])

z,x = np.loadtxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\XYZT_tip_0Hz3.txt")
x3 = x[2000:]
peaks3 = scipy.signal.find_peaks(x3,height=[0,0.5],distance=10)[0][:-1]
peaks3_n = scipy.signal.find_peaks(x3,height=[-0.5,0],distance=10)[0][:-1]
plt.plot(x3[peaks3])
plt.show()
b3 = np.polyfit(peaks3/150,np.log(x3[peaks3]),1)
plt.plot(peaks3/150,f(peaks3/150,b3,1))
plt.plot(peaks3/150,np.log(x3[peaks3]))
plt.show()
fft3 = np.fft.rfft(x3)
plt.plot(x3)
plt.plot(peaks3,x3[peaks3])
plt.plot(peaks3_n,x3[peaks3_n])
plt.show()
plt.plot((np.real(fft3[:200])**2+np.imag(fft3[:200])**2)**0.5)
plt.show()
f3 = 150*(peaks3.shape[0]-1)/(peaks3[-1]-peaks3[0])
