import numpy as np
import matplotlib.pyplot as plt

# Number of beam mode shapes used in the Galerkin decomposition
N = 10

# Lenght of the pipe
L = 0.46

def Eq_Cara(Lambda):
    return np.cos(Lambda)*np.cosh(Lambda)+1

LAMBDA = []

for i in range(N):
    LL_Guess = np.pi*(2*i+1)/2
    x0 = LL_Guess + 0.1
    x1 = LL_Guess - 0.1
    while abs(x0-x1)>10**-16:
        xnew = x0 - (x0-x1)*Eq_Cara(x0)/(Eq_Cara(x0)-Eq_Cara(x1))
        x1 = x0
        x0 = xnew
    LAMBDA.append(x0)

def sigma(r):
    return ((np.sinh(LAMBDA[r])-np.sin(LAMBDA[r]))/(np.cosh(LAMBDA[r])+np.cos(LAMBDA[r])))

def phi(r,x):
    return np.cosh(x*LAMBDA[r])-np.cos(x*LAMBDA[r])-sigma(r)*(np.sinh(x*LAMBDA[r])-np.sin(x*LAMBDA[r]))

def bsr(s,r):
    if s == r:
        return 2
    else:
        return 4/((LAMBDA[s]/LAMBDA[r])**2+(-1)**(r+s))
    
def csr(s,r):
    if s == r:
        return LAMBDA[r]*sigma(r)*(2-LAMBDA[r]*sigma(r))
    else:
        return 4*(LAMBDA[r]*sigma(r)-LAMBDA[s]*sigma(s))/((-1)**(r+s)-(LAMBDA[s]/LAMBDA[r])**2)
    
def dsr(s,r):
    if s == r:
        return csr(s,r)/2
    else:
        return (4*(LAMBDA[r]*sigma(r)-LAMBDA[s]*sigma(s)+2)*(-1)**(r+s))/(1-(LAMBDA[s]/LAMBDA[r])**4)-((3+(LAMBDA[s]/LAMBDA[r])**4)/(1-((LAMBDA[s]/LAMBDA[r])**4)))*bsr(s,r)                                                                                                                                               

B = np.zeros((N,N))
C = np.zeros((N,N))
D = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        B[i,j] = bsr(i,j)  
        C[i,j] = csr(i,j)
        D[i,j] = dsr(i,j)

Delta = np.zeros((N,N))
for i in range(N):
    Delta[i,i] = LAMBDA[i]**4

MM = np.eye(N)

u = 0
beta = 0
alpha_array = np.array([0.005806])
eta_s = 0.01

N_vect = np.linspace(0,N-1,N)
phi_vect = np.zeros(N)
for k in range(N):
    phi_vect[k] = phi(k,L)

gamma = 355.04
for a in range(len(alpha_array)):
    alpha = alpha_array[a]
    F = np.zeros((N,N))
    for i in range(N):
        F[i,i] = alpha*LAMBDA[i]**4
    C_g = 2*(beta**0.5)*u*B + F
    K = Delta + gamma*B + (u**2-gamma)*C + gamma*D
    
    S = np.zeros(N)
    for r in range(N):
        S[r] = (np.sinh(LAMBDA[r])-np.sin(LAMBDA[r])-sigma(r)*(np.cosh(LAMBDA[r])+np.cos(LAMBDA[r])-2))
    
    y_max = np.zeros(500)
    omega_s_array = np.linspace(3*2*np.pi,2*np.pi*20,501)
    for o in range(1,len(omega_s_array)):
        omega_s = omega_s_array[o]
        Qe = eta_s*omega_s**2*np.dot(np.linalg.inv(-omega_s**2*MM+1j*omega_s*C_g+K),S)
        Q = (np.real(Qe)**2+np.imag(Qe)**2)**0.5
        teta = np.arccos(np.real(Qe)/Q)
        y = np.zeros((101,N))
        tau_array = np.linspace(0,2*np.pi/omega_s,101)
        for t in range(101):
            tau = tau_array[t]
            for k in range(N):
                y[t,k] = phi(k,L)*Q[k]*np.cos(omega_s*tau+teta[k])
        y_max[o-1] = max(np.sum(y,axis=1))
    plt.plot(omega_s_array[1:]/(2*np.pi),y_max,color=(0.3,0.3,0.3))
plt.title("Amplitude against the shaker frequency for Gamma = "+str(gamma))
plt.xlabel("Frequency from the shaker in Hz")
plt.ylabel("Amplitude of the tip in meters")
plt.show()
    
