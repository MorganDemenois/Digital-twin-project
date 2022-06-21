# =============================================================================
# This code is used to plot the evolution of the damping with respect to the 
# flowrate value in the pipe
# =============================================================================

# Loading of the librairies
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# Non dimensional parameters of the pipe
d_array = np.array([6.35*10**-3,6.35*10**-3,6.35*10**-3])
D_array = np.array([15.875*10**-3,15.875*10**-3,15.875*10**-3])
M_silicone = 1.34/1000*100**3
m_array = M_silicone*np.pi*(D_array**2-d_array**2)/4
M_array = 996*np.pi*d_array**2/4
I_array = np.pi*(D_array**4-d_array**4)/64
L_array = np.array([46*10**-2,41*10**-2,36*10**-2])
g = 9.81
Estar = 5333.901583749264
E = 225171.9440703812
alpha_array = (I_array/(E*(M_array+m_array)))**0.5*Estar/L_array**2
beta_array = M_array/(M_array+m_array)
gamma_array = (M_array+m_array)*L_array**3*g/(E*I_array)

# Number of beam mode shapes used in the Galerkin decomposition
N = 10

# Equation to solve to obtain the beam eigenfrequencies
def Eq_Cara(Lambda):
    return np.cos(Lambda)*np.cosh(Lambda)+1

# Use of the Newton method to solve the previous equation
LAMBDA = []
for i in range(N):
    LL_Guess = np.pi*(2*i+1)/2 # Initialization of the eigenfrequencies at an approached value
    x0 = LL_Guess + 0.1 
    x1 = LL_Guess - 0.1
    while abs(x0-x1)>10**-16: # Loop of the Newton solver with the termination criterion
        xnew = x0 - (x0-x1)*Eq_Cara(x0)/(Eq_Cara(x0)-Eq_Cara(x1))
        x1 = x0
        x0 = xnew
    LAMBDA.append(x0)
    
# Function used to build the beam mode shapes
def sigma(r):
    return ((np.sinh(LAMBDA[r])-np.sin(LAMBDA[r]))/(np.cosh(LAMBDA[r])+np.cos(LAMBDA[r])))

# Functions used to build the B, C and D matrices
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

# Construction of the B, C and D matrices
B = np.zeros((N,N))
C = np.zeros((N,N))
D = np.zeros((N,N))
Mi = np.eye(N)
for i in range(N):
    for j in range(N):
        B[i,j] = bsr(i,j)  
        C[i,j] = csr(i,j)
        D[i,j] = dsr(i,j)

# Construction of the diagonal matrix
Delta = np.zeros((N,N))
for i in range(N):
    Delta[i,i] = LAMBDA[i]**4

# This function returns the eigenvalues and eigenvectors of the time modes with respect to the flowrate u
def result(u,alpha,beta,gamma):
    FF = np.zeros((N,N))
    for i in range(N):
        FF[i,i] = alpha*LAMBDA[i]**4
    S = 2*(beta**0.5)*u*B + FF # Construction of the damping matrix
    K = Delta + (u**2-gamma)*C + gamma*B + gamma*D # Construction of the stifness matrix
    F = np.block([[np.zeros((N,N)),Mi],[Mi,S]]) # Reduced order matrix
    E = np.block([[-Mi,np.zeros((N,N))],[np.zeros((N,N)),K]]) # Reduced order matrix  
    eigenValues, eigenVectors = np.linalg.eig(-np.dot(np.linalg.inv(F),E)) # Solving of the eigenvalue problem
    return eigenValues, eigenVectors 

# This function returns the deflection of the pipe for the (x,t) coordinates and for a given flowrate
def eta(x,t,u):
    # Compute the eigenvalues of the times functions
    eigenValues, eigenVectors = result(u,beta,gamma)
    # Use of the initial conditions to compute the constant parameters of the solution
    # At t=0 the deflection of the pipe is supposed to be the 0.3x the first beam mode shape
    CL = np.zeros(2*N)
    CL[0] = 0.3
    Constant = np.dot(np.linalg.inv(eigenVectors),CL)
    Sol = 0
    for r in range(N):
        qr = 0
        for i in range(2*N):
            qr += Constant[i]*np.exp(eigenValues[i]*t)*eigenVectors[r,i]
        phir = np.cosh(LAMBDA[r]*x)-np.cos(LAMBDA[r]*x)-sigma(r)*(np.sinh(LAMBDA[r]*x)-np.sin(LAMBDA[r]*x))
        Sol += qr*phir
    return Sol

U_array_3 = np.zeros((3,1000))
log_array = np.zeros((3,1000))
# Loop on the 3 different cases
for i in range(3):
    alpha = alpha_array[i]
    beta = beta_array[i]
    gamma = gamma_array[i]
    d = d_array[i]
    M = M_array[i]
    m = m_array[i]
    I = I_array[i]
    log_dec = []
    L = L_array[i]
    # Computation of the critical speed (upper limit for the flowrate)
    ucr = 0
    u_array = np.linspace(0,100,10001)
    for j in range(len(u_array)):
        u = u_array[j]
        if ucr == 0 and max(np.real(result(u, alpha, beta, gamma)[0])) >= 0:
            ucr = u_array[j-1]
    u_array = np.linspace(0,ucr,1000)
    U_array = u_array*(E*I/M)**0.5/L
    tau = np.linspace(0,2,200)
    t = tau*((M+m)/(E*I))**0.5*L**2
    # Loop on the flowrate from 0 to ucr
    for u in u_array:
        # Computation of the deflection of the tip with respect to time
        A = L*eta(1,tau,u,alpha,beta,gamma)
        # Find the peaks of the deflection corresponding to the amplitude of the deflection
        peaks_no0 = scipy.signal.find_peaks(abs(A),height=[0,2],distance=10)[0][:-1]
        peaks = np.array([0 for i in range(peaks_no0.shape[0]+1)])
        peaks[1:] = peaks_no0
        # Fit a first order polynomial function to the logarithm of the amplitude
        b = np.polyfit(t[peaks],np.log(abs(A[peaks])),1)
        # The first coefficient of the polynomial corresponds to the decrement (ie damping)
        log_dec.append(b[0])
    U_array_3[i] = U_array
    log_array[i] = log_dec

# Plotting of the results
for i in range(3):
    plt.plot(U_array_3[i],log_array[i],label=str(i))
plt.legend()
plt.show()

# Saving of the results
np.savetxt(r"C:\Users\Morgan\Videos\Theoritical_Damping_Evolution_3L_Damping.txt",log_array)
np.savetxt(r"C:\Users\Morgan\Videos\Theoritical_Damping_Evolution_3L_U.txt",U_array_3)