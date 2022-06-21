import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import tikzplotlib as tpl
plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Computer Modern Roman"]})
plt.rcParams['figure.autolayout'] = True

d = 6.35*10**-3

files =['26.5','27','27.5','28','28.5','29','29.5','30','30.5','31','31.5','32','32.5','33',
  '33.5','34','34.5','35','35.5','36','36.5','37','37.5','38','38.5','39','39.5','40','40.5','41',
  '41.5','42','42.5','43','43.5','44','44.5','45','45.5','46','46.5','47','47.5','48','48.5','49',
  '49.5','50','50.5','51','51.5','52','52.5','53','53.5','54','54.5','55','55.5','56','56.5','57',
  '57.5','58','58.5','59','59.5','60'] 

# files = ["28.5","42","60"]

weighted_freq = []
maximum_amplitude = []
maximum_freq = []
front_freq = []
side_freq = []
flowrate = []
for file in files:
    U_array = np.loadtxt(r"C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\U_"+file+"Hz.txt")
    Uv = np.mean(U_array)
    print(Uv)
    U = round(Uv*4/(60*1000*np.pi*d**2),3)
    flowrate.append(U)
    xyzt = np.loadtxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\XYZT_tip_"+file+"Hz.txt")
    # plt.plot(np.linspace(0,300/150,300),((xyzt[0]**2+xyzt[1]**2)**0.5)[:300])
    # plt.title("Amplitude of the tip of the pipe for U = "+str(U)+"kg/s")
    # plt.xlabel("Time in seconds")
    # plt.ylabel("Amplitude in meters")
    # plt.show()
    # plt.plot(xyzt[0],xyzt[1],color=(0.2,0.2,0.2))
    # plt.title("D=0.25 L = 0.46 Trace of the tip of the pipe for file = "+str(file))
    # plt.xlabel("Amplitude along the x axis in meters")
    # plt.ylabel("Amplitude along the y axis in meters")
    # # plt.savefig("Trace"+str(U)+".jpg")
    # tpl.save(r"C:\Users\Morgan\Desktop\Trace_"+str(file)+"Hz_D=0.25_L=0.46.tex")
    # plt.show()
    fft = np.fft.rfft(((xyzt[0]**2+xyzt[1]**2)**0.5))
    # plt.plot(np.linspace(150/4500,150*1000/4500,999),((np.real(fft)**2+np.imag(fft)**2)**0.5)[1:1000])
    # plt.title("FFT of the tip of the pipe for U = "+str(U)+"m/s")
    # plt.xlabel("Frequence in Hz")
    # plt.ylabel("Amplitude")
    # plt.show()
    fft_front = np.fft.rfft(xyzt[0])
    fft_side = np.fft.rfft(xyzt[1])
    front_freq.append(np.linspace(150/4500,150*1000/4500,999)[np.argmax(((np.real(fft_front)**2+np.imag(fft_front)**2)**0.5)[1:1000])])
    side_freq.append(np.linspace(150/4500,150*1000/4500,999)[np.argmax(((np.real(fft_side)**2+np.imag(fft_side)**2)**0.5)[1:1000])])
    maximum_freq.append(np.linspace(150/4500,150*1000/4500,999)[np.argmax(((np.real(fft)**2+np.imag(fft)**2)**0.5)[1:1000])])
    maximum_amplitude.append(np.mean(-np.partition(-((xyzt[0]**2+xyzt[1]**2)**0.5),20)[:20]))
    weighted_freq.append(np.average(np.linspace(150/4500,150*1000/4500,999),weights=((np.real(fft)**2+np.imag(fft)**2)**0.5)[1:1000]))

# plt.plot(flowrate,maximum_amplitude,color=(0.3,0.3,0.3))
# plt.title("Evolution of the amplitude of the tip with the flowrate")
# plt.xlabel("Flowrate in m/s")
# plt.ylabel("Amplitude in meters")
# # plt.savefig("A(u).jpg")
# plt.show()

# # plt.plot(flowrate[1:],maximum_freq[1:])
# # plt.title("Evolution of the frequency of the first mode with the flowrate")
# # plt.xlabel("Flowrate in m/s")
# # plt.ylabel("Frequency in Hz")
# # plt.show()

# # plt.plot(flowrate[1:],weighted_freq[1:])
# # plt.title("Evolution of the weighted frequency with the flowrate")
# # plt.xlabel("Flowrate in m/s")
# # plt.ylabel("Frequency in Hz")
# # plt.show()

# # plt.plot(flowrate[1:],front_freq[1:],label="front")
# # plt.plot(flowrate[1:],side_freq[1:],label="side")
# plt.plot(flowrate[:],(np.array(front_freq[:])+np.array(side_freq[:]))/2,color=(0.3,0.3,0.3))
# plt.title("Evolution of the pipe's frequency with the flowrate")
# plt.xlabel("Flowrate in m/s")
# plt.ylabel("Frequency in Hz")
# # plt.savefig('f(u).jpg')
# plt.show()

flowrate_case_1 = flowrate
amplitude_case_1 = maximum_amplitude
frequency_case_1 = (np.array(front_freq[:])+np.array(side_freq[:]))/2

d = 6.35*10**-3

files = ["27.5","28","28.5","29","29.5","30","31","32","33","34","35","36","37","38","39",\
          "40","41","42","43","44","45","46","47","48","49","50","51","52","53","54","55",\
          "56","57","58","58.5","59","59.5","60"]

# files = ["50"]

weighted_freq = []
maximum_amplitude = []
maximum_freq = []
front_freq = []
side_freq = []
flowrate = []
for file in files:
    U_array = np.loadtxt(r"C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_41cm\U_"+file+"Hz.txt")
    Uv = np.mean(U_array)
    print(Uv)
    U = round(Uv*4/(60*1000*np.pi*d**2),3)
    flowrate.append(U)
    xyzt = np.loadtxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_41cm\XYZT_tip_"+file+"Hz.txt")
    # plt.plot(np.linspace(0,300/150,300),((xyzt[0]**2+xyzt[1]**2)**0.5)[:300])
    # plt.title("Amplitude of the tip of the pipe for U = "+str(U)+"kg/s")
    # plt.xlabel("Time in seconds")
    # plt.ylabel("Amplitude in meters")
    # plt.show()
    # plt.plot(xyzt[0],xyzt[1],color=(0.2,0.2,0.2))
    # plt.title("D=0.25 L = 0.41 Trace of the tip of the pipe for file = "+str(file))
    # plt.xlabel("Amplitude along the x axis in meters")
    # plt.ylabel("Amplitude along the y axis in meters")
    # tpl.save(r"C:\Users\Morgan\Desktop\Trace_"+str(file)+"Hz_D=0.25_L=0.41.tex")
    # plt.show()
    fft = np.fft.rfft(((xyzt[0]**2+xyzt[1]**2)**0.5))
    # plt.plot(np.linspace(150/4500,150*1000/4500,999),((np.real(fft)**2+np.imag(fft)**2)**0.5)[1:1000])
    # plt.title("FFT of the tip of the pipe for U = "+str(U)+"m/s")
    # plt.xlabel("Frequence in Hz")
    # plt.ylabel("Amplitude")
    # plt.show()
    fft_front = np.fft.rfft(xyzt[0])
    fft_side = np.fft.rfft(xyzt[1])
    front_freq.append(np.linspace(150/4500,150*1000/4500,999)[np.argmax(((np.real(fft_front)**2+np.imag(fft_front)**2)**0.5)[1:1000])])
    side_freq.append(np.linspace(150/4500,150*1000/4500,999)[np.argmax(((np.real(fft_side)**2+np.imag(fft_side)**2)**0.5)[1:1000])])
    maximum_freq.append(np.linspace(150/4500,150*1000/4500,999)[np.argmax(((np.real(fft)**2+np.imag(fft)**2)**0.5)[1:1000])])
    maximum_amplitude.append(np.mean(-np.partition(-((xyzt[0]**2+xyzt[1]**2)**0.5),20)[:20]))
    weighted_freq.append(np.average(np.linspace(150/4500,150*1000/4500,999),weights=((np.real(fft)**2+np.imag(fft)**2)**0.5)[1:1000]))

# plt.plot(flowrate,maximum_amplitude,color=(0.3,0.3,0.3))
# plt.title("Evolution of the amplitude of the tip with the flowrate")
# plt.xlabel("Flowrate in m/s")
# plt.ylabel("Amplitude in meters")
# plt.show()

# # plt.plot(flowrate[1:],maximum_freq[1:])
# # plt.title("Evolution of the frequency of the first mode with the flowrate")
# # plt.xlabel("Flowrate in m/s")
# # plt.ylabel("Frequency in Hz")
# # plt.show()

# # plt.plot(flowrate[1:],weighted_freq[1:])
# # plt.title("Evolution of the weighted frequency with the flowrate")
# # plt.xlabel("Flowrate in m/s")
# # plt.ylabel("Frequency in Hz")
# # plt.show()

# # plt.plot(flowrate[1:],front_freq[1:],label="front")
# # plt.plot(flowrate[1:],side_freq[1:],label="side")
# plt.plot(flowrate[:],(np.array(front_freq[:])+np.array(side_freq[:]))/2,color=(0.3,0.3,0.3))
# plt.title("Evolution of the pipe's frequency with the flowrate")
# plt.xlabel("Flowrate in m/s")
# plt.ylabel("Frequency in Hz")
# # plt.savefig('f(u).jpg')
# plt.show()

flowrate_case_2 = flowrate
amplitude_case_2 = maximum_amplitude
frequency_case_2 = (np.array(front_freq[:])+np.array(side_freq[:]))/2

d = 6.35*10**-3

files = ["27.5","28","28.5","29","29.5","30","31","32","33","34","35","36","37","38","39",\
          "40","41","42","43","44","45","46","47","48","49","50","51","52","53","54","55",\
          "56","57","58","58.5","59","59.5","60"]

# files = []

weighted_freq = []
maximum_amplitude = []
maximum_freq = []
front_freq = []
side_freq = []
flowrate = []
for file in files:
    U_array = np.loadtxt(r"C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_36cm\U_"+file+"Hz.txt")
    Uv = np.mean(U_array)
    print(Uv)
    U = round(Uv*4/(60*1000*np.pi*d**2),3)
    flowrate.append(U)
    xyzt = np.loadtxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_36cm\XYZT_tip_"+file+"Hz.txt")
    # plt.plot(np.linspace(0,300/150,300),((xyzt[0]**2+xyzt[1]**2)**0.5)[:300])
    # plt.title("Amplitude of the tip of the pipe for U = "+str(U)+"kg/s")
    # plt.xlabel("Time in seconds")
    # plt.ylabel("Amplitude in meters")
    # plt.show()
    # plt.plot(xyzt[0],xyzt[1],color=(0.2,0.2,0.2))
    # plt.title("D=0.25 L = 0.36 Trace of the tip of the pipe for file = "+str(file))
    # plt.xlabel("Amplitude along the x axis in meters")
    # plt.ylabel("Amplitude along the y axis in meters")
    # tpl.save(r"C:\Users\Morgan\Desktop\Trace_"+str(file)+"Hz_D=0.25_L=0.36.tex")
    # plt.show()
    fft = np.fft.rfft(((xyzt[0]**2+xyzt[1]**2)**0.5))
    # plt.plot(np.linspace(150/4500,150*1000/4500,999),((np.real(fft)**2+np.imag(fft)**2)**0.5)[1:1000])
    # plt.title("FFT of the tip of the pipe for U = "+str(U)+"m/s")
    # plt.xlabel("Frequence in Hz")
    # plt.ylabel("Amplitude")
    # plt.show()
    fft_front = np.fft.rfft(xyzt[0])
    fft_side = np.fft.rfft(xyzt[1])
    front_freq.append(np.linspace(150/4500,150*1000/4500,999)[np.argmax(((np.real(fft_front)**2+np.imag(fft_front)**2)**0.5)[1:1000])])
    side_freq.append(np.linspace(150/4500,150*1000/4500,999)[np.argmax(((np.real(fft_side)**2+np.imag(fft_side)**2)**0.5)[1:1000])])
    maximum_freq.append(np.linspace(150/4500,150*1000/4500,999)[np.argmax(((np.real(fft)**2+np.imag(fft)**2)**0.5)[1:1000])])
    maximum_amplitude.append(np.mean(-np.partition(-((xyzt[0]**2+xyzt[1]**2)**0.5),20)[:20]))
    weighted_freq.append(np.average(np.linspace(150/4500,150*1000/4500,999),weights=((np.real(fft)**2+np.imag(fft)**2)**0.5)[1:1000]))

# plt.plot(flowrate,maximum_amplitude,color=(0.3,0.3,0.3))
# plt.title("Evolution of the amplitude of the tip with the flowrate")
# plt.xlabel("Flowrate in m/s")
# plt.ylabel("Amplitude in meters")
# plt.show()

# # plt.plot(flowrate[1:],maximum_freq[1:])
# # plt.title("Evolution of the frequency of the first mode with the flowrate")
# # plt.xlabel("Flowrate in m/s")
# # plt.ylabel("Frequency in Hz")
# # plt.show()

# # plt.plot(flowrate[1:],weighted_freq[1:])
# # plt.title("Evolution of the weighted frequency with the flowrate")
# # plt.xlabel("Flowrate in m/s")
# # plt.ylabel("Frequency in Hz")
# # plt.show()

# # plt.plot(flowrate[1:],front_freq[1:],label="front")
# # plt.plot(flowrate[1:],side_freq[1:],label="side")
# plt.plot(flowrate[:],(np.array(front_freq[:])+np.array(side_freq[:]))/2,color=(0.3,0.3,0.3))
# plt.title("Evolution of the pipe's frequency with the flowrate")
# plt.xlabel("Flowrate in m/s")
# plt.ylabel("Frequency in Hz")
# # plt.savefig('f(u).jpg')
# plt.show()

flowrate_case_3 = flowrate
amplitude_case_3 = maximum_amplitude
frequency_case_3 = (np.array(front_freq[:])+np.array(side_freq[:]))/2

d = 7.9375*10**-3

files = ["21","21.5","22","23","24","25","26","27","28","29","30","31","32","33",\
          "34","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","52","53",\
          "54","55","56","57","58"] 

# files = ["22"]

maximum_amplitude = []
maximum_freq = []
weighted_freq = []
flowrate = []
front_freq = []
side_freq = []
for file in files:
    U_array = np.loadtxt(r"C:\\Users\Morgan\Videos\Videos_Pipe_0.3125in_46cm\U_"+file+"Hz.txt")
    Uv = np.mean(U_array)
    print(Uv)
    U = round(Uv*4/(60*1000*np.pi*d**2),3)
    flowrate.append(U)
    xyzt = np.loadtxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.3125in_46cm\XYZT_tip_"+file+"Hz.txt")
    # plt.plot(np.linspace(0,300/150,300),((xyzt[0]**2+xyzt[1]**2)**0.5)[:300])
    # plt.title("Amplitude of the tip of the pipe for U = "+str(U)+"kg/s")
    # plt.xlabel("Time in seconds")
    # plt.ylabel("Amplitude in meters")
    # plt.show()
    # plt.plot(np.linspace(0,300/150,300),xyzt[0][:300])
    # plt.title("x Amplitude of the tip of the pipe for U = "+str(U)+"kg/s and file = "+str(file))
    # plt.xlabel("Time in seconds")
    # plt.ylabel("Amplitude in meters")
    # plt.show()
    # plt.plot(np.linspace(0,300/150,300),xyzt[1][:300])
    # plt.title("y Amplitude of the tip of the pipe for U = "+str(U)+"kg/s and file = "+str(file))
    # plt.xlabel("Time in seconds")
    # plt.ylabel("Amplitude in meters")
    # plt.show()
    # plt.plot(xyzt[0],xyzt[1],color=(0.2,0.2,0.2))
    # plt.title("D=0.3125 L = 0.46 Trace of the tip of the pipe for file = "+str(file))
    # plt.xlabel("Amplitude along the x axis in meters")
    # plt.ylabel("Amplitude along the y axis in meters")
    # tpl.save(r"C:\Users\Morgan\Desktop\Trace_"+str(file)+"Hz_D=0.3125_L=0.46.tex")
    # plt.show()
    fft = np.fft.rfft(((xyzt[0]**2+xyzt[1]**2)**0.5))
    # plt.plot(np.linspace(150/4500,150*1000/4500,999),((np.real(fft)**2+np.imag(fft)**2)**0.5)[1:1000])
    # plt.title("FFT of the tip of the pipe for U = "+str(U)+"m/s")
    # plt.xlabel("Frequence in Hz")
    # plt.ylabel("Amplitude")
    # plt.show()
    fft_front = np.fft.rfft(xyzt[0])
    fft_side = np.fft.rfft(xyzt[1])
    front_freq.append(np.linspace(150/4500,150*1000/4500,999)[np.argmax(((np.real(fft_front)**2+np.imag(fft_front)**2)**0.5)[1:1000])])
    side_freq.append(np.linspace(150/4500,150*1000/4500,999)[np.argmax(((np.real(fft_side)**2+np.imag(fft_side)**2)**0.5)[1:1000])])
    maximum_freq.append(np.linspace(150/4500,150*1000/4500,999)[np.argmax(((np.real(fft)**2+np.imag(fft)**2)**0.5)[1:1000])])
    weighted_freq.append(np.average(np.linspace(150/4500,150*1000/4500,999),weights=((np.real(fft)**2+np.imag(fft)**2)**0.5)[1:1000]))
    maximum_amplitude.append(np.mean(-np.partition(-((xyzt[0]**2+xyzt[1]**2)**0.5),20)[:20]))
    
# plt.plot(flowrate,maximum_amplitude,color=(0.3,0.3,0.3))
# plt.title("Evolution of the amplitude of the tip with the flowrate")
# plt.xlabel("Flowrate in m/s")
# plt.ylabel("Amplitude in meters")
# plt.show()

# # plt.plot(flowrate[1:],maximum_freq[1:])
# # plt.title("Evolution of the frequency of the first mode with the flowrate")
# # plt.xlabel("Flowrate in m/s")
# # plt.ylabel("Frequency in Hz")
# # plt.show()

# # plt.plot(flowrate[1:],weighted_freq[1:])
# # plt.title("Evolution of the weighted frequency with the flowrate")
# # plt.xlabel("Flowrate in m/s")
# # plt.ylabel("Frequency in Hz")
# # plt.show()

# # plt.plot(flowrate[1:],front_freq[1:],label="front")
# # plt.plot(flowrate[1:],side_freq[1:],label="side")
# plt.plot(flowrate[:],(np.array(front_freq[:])+np.array(side_freq[:]))/2,color=(0.3,0.3,0.3))
# plt.title("Evolution of the pipe's frequency with the flowrate")
# plt.xlabel("Flowrate in m/s")
# plt.ylabel("Frequency in Hz")
# # plt.savefig('f(u).jpg')
# plt.show()

flowrate_case_4 = flowrate
amplitude_case_4 = maximum_amplitude
frequency_case_4 = (np.array(front_freq[:])+np.array(side_freq[:]))/2

d = 9.525*10**-3

files = ["22","22.5","23","23.5","24","25","26","27","28","29","30","31","32","33","34","35",\
          "36","37","38","39","40","41","42"]

# files = ["29"]

weighted_freq = []
maximum_amplitude = []
maximum_freq = []
flowrate = []
front_freq = []
side_freq = []
for file in files:
    U_array = np.loadtxt(r"C:\\Users\Morgan\Videos\Videos_Pipe_0.365in_46cm\U_"+file+"Hz.txt")
    Uv = np.mean(U_array)
    print(Uv)
    U = round(Uv*4/(60*1000*np.pi*d**2),3)
    flowrate.append(U)
    xyzt = np.loadtxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.365in_46cm\XYZT_tip_"+file+"Hz.txt")

    plt.plot(np.linspace(0,4500/150,4500),xyzt[0][:4500])
    plt.title("x amplitude file = "+str(file))
    plt.show()
    plt.plot(np.linspace(0,4500/150,4500),+xyzt[1][:4500])
    plt.title("y amplitude file = "+str(file))
    plt.show()
    
    # plt.plot(np.linspace(0,300/150,300),((xyzt[0]**2+xyzt[1]**2)**0.5)[:300])
    # plt.title("Amplitude of the tip of the pipe for U = "+str(U)+"kg/s")
    # plt.xlabel("Time in seconds")
    # plt.ylabel("Amplitude in meters")
    # plt.show()
    # plt.plot(xyzt[0],xyzt[1],color=(0.2,0.2,0.2))
    # plt.title("D=0.365 L = 0.46 Trace of the tip of the pipe for file = "+str(file))
    # plt.xlabel("Amplitude along the x axis in meters")
    # plt.ylabel("Amplitude along the y axis in meters")
    # tpl.save(r"C:\Users\Morgan\Desktop\Trace_"+str(file)+"Hz_D=0.365_L=0.46.tex")
    # plt.show()
    fft = np.fft.rfft(((xyzt[0]**2+xyzt[1]**2)**0.5))
    # plt.plot(np.linspace(150/4500,150*1000/4500,999),((np.real(fft)**2+np.imag(fft)**2)**0.5)[1:1000])
    # plt.title("FFT of the tip of the pipe for U = "+str(U)+"m/s")
    # plt.xlabel("Frequence in Hz")
    # plt.ylabel("Amplitude")
    # plt.show()
    fft_front = np.fft.rfft(xyzt[0])
    fft_side = np.fft.rfft(xyzt[1])
    front_freq.append(np.linspace(150/4500,150*1000/4500,999)[np.argmax(((np.real(fft_front)**2+np.imag(fft_front)**2)**0.5)[1:1000])])
    side_freq.append(np.linspace(150/4500,150*1000/4500,999)[np.argmax(((np.real(fft_side)**2+np.imag(fft_side)**2)**0.5)[1:1000])])    
    maximum_freq.append(np.linspace(150/4500,150*1000/4500,999)[np.argmax(((np.real(fft)**2+np.imag(fft)**2)**0.5)[1:1000])])
    maximum_amplitude.append(np.mean(-np.partition(-((xyzt[0]**2+xyzt[1]**2)**0.5),20)[:20]))
    weighted_freq.append(np.average(np.linspace(150/4500,150*1000/4500,999),weights=((np.real(fft)**2+np.imag(fft)**2)**0.5)[1:1000]))

# plt.plot(flowrate[:-1],maximum_amplitude[:-1],color=(0.3,0.3,0.3))
# plt.title("Evolution of the amplitude of the tip with the flowrate")
# plt.xlabel("Flowrate in m/s")
# plt.ylabel("Amplitude in meters")
# plt.show()

# plt.plot(flowrate[1:-1],maximum_freq[1:-1])
# plt.title("Evolution of the frequency of the first mode with the flowrate")
# plt.xlabel("Flowrate in m/s")
# plt.ylabel("Frequency in Hz")
# plt.show()

# plt.plot(flowrate[1:-1],weighted_freq[1:-1])
# plt.title("Evolution of the weighted frequency with the flowrate")
# plt.xlabel("Flowrate in m/s")
# plt.ylabel("Frequency in Hz")
# plt.show()

# # plt.plot(flowrate[1:],front_freq[1:],label="front")
# # plt.plot(flowrate[1:],side_freq[1:],label="side")
# plt.plot(flowrate[:-1],(np.array(front_freq[:-1])+np.array(side_freq[:-1]))/2,color=(0.3,0.3,0.3))
# plt.title("Evolution of the pipe's frequency with the flowrate")
# plt.xlabel("Flowrate in m/s")
# plt.ylabel("Frequency in Hz")
# # plt.savefig('f(u).jpg')
# plt.show()

flowrate_case_5 = flowrate[:-1]
amplitude_case_5 = maximum_amplitude[:-1]
frequency_case_5 = (np.array(front_freq[:-1])+np.array(side_freq[:-1]))/2

plt.plot(flowrate_case_1,amplitude_case_1,color=(0.3,0.3,0.3),label="L=46cm")
plt.plot(flowrate_case_2,amplitude_case_2,color=(0.3,0.3,0.3),label="L=41cm",linestyle="dashed")
plt.plot(flowrate_case_3,amplitude_case_3,color=(0.3,0.3,0.3),label="L=36cm",linestyle="dotted")
plt.xlabel("Flowrate in m/s")
plt.ylabel("Amplitude in meters")
plt.legend()
# tpl.save(r"C:\Users\Morgan\Desktop\Amplitude_3D.tex")
plt.show()

plt.plot(flowrate_case_1,amplitude_case_1,color=(0.3,0.3,0.3),label="d=0.25in")
plt.plot(flowrate_case_4,amplitude_case_4,color=(0.3,0.3,0.3),label="d=0.3125in",linestyle="dashed")
plt.plot(flowrate_case_5,amplitude_case_5,color=(0.3,0.3,0.3),label="d=0.365in",linestyle="dotted")
plt.xlabel("Flowrate in m/s")
plt.ylabel("Amplitude in meters")
plt.legend()
# tpl.save(r"C:\Users\Morgan\Desktop\Amplitude_3L.tex")
plt.show()

# plt.plot(flowrate_case_1,frequency_case_1,color=(0.3,0.3,0.3),label="L=46cm")
# plt.plot(flowrate_case_2,frequency_case_2,color=(0.3,0.3,0.3),label="L=41cm",linestyle="dashed")
# plt.plot(flowrate_case_3,frequency_case_3,color=(0.3,0.3,0.3),label="L=36cm",linestyle="dotted")
# plt.xlabel("Flowrate in m/s")
# plt.ylabel("Frequency in Hz")
# plt.legend()
# # tpl.save(r"C:\Users\Morgan\Desktop\Frequency_3D.tex")
# plt.show()

# plt.plot(flowrate_case_1,frequency_case_1,color=(0.3,0.3,0.3),label="d=0.25in")
# plt.plot(flowrate_case_4,frequency_case_4,color=(0.3,0.3,0.3),label="d=0.3125in",linestyle="dashed")
# plt.plot(flowrate_case_5,frequency_case_5,color=(0.3,0.3,0.3),label="d=0.365in",linestyle="dotted")
# plt.xlabel("FLowrate in m/s")
# plt.ylabel("Frequency in Hz")
# plt.legend()
# # tpl.save(r"C:\Users\Morgan\Desktop\Frequency_3L.tex")
# plt.show()