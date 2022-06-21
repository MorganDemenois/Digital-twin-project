# =============================================================================
# This code is used to plot the evolution of the critical speed with respect to
# the gamma and beta parameters
# =============================================================================

# Loading of the librairies
import numpy as np
import matplotlib.pyplot as plt

# Non dimensional parameters of the pipe
L = 1
array_beta = np.linspace(0,1,101)  
array_gamma = np.array([-10,0,10,100])  
u_array = np.linspace(0,50,501)  
alpha = 0

# Step of evolution of the flowrate
du = 0.02

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
M = np.eye(N)
for i in range(N):
    for j in range(N):
        B[i,j] = bsr(i,j)  
        C[i,j] = csr(i,j)
        D[i,j] = dsr(i,j)

# Construction of the diagonal matrices
Delta = np.zeros((N,N))
FF = np.zeros((N,N))
for i in range(N):
    Delta[i,i] = LAMBDA[i]**4
    FF[i,i] = alpha*LAMBDA[i]**4

# This function returns the eigenvalues and eigenvectors of the time modes with respect to the flowrate u
def result(u,beta,gamma):
    S = 2*beta**0.5*u*B # Construction of the damping matrix
    K = Delta + gamma*B + (u**2-gamma)*C + gamma*D # Construction of the stifness matrix
    F = np.block([[np.zeros((N,N)),M],[M,S]]) # Reduced order matrix
    E = np.block([[-M,np.zeros((N,N))],[np.zeros((N,N)),K]]) # Reduced order matrix
    eigenValues, eigenVectors = np.linalg.eig(-np.dot(np.linalg.inv(F),E)) # Solving of the eigenvalue problem
    return eigenValues, eigenVectors

b_critique = np.zeros((len(array_gamma),len(u_array)))

# Loop on the gamma values
for g in range(len(array_gamma)):  
    
    # Loop on the flowrate
    for i in range(len(u_array)):  
        u = u_array[i]
        gamma = array_gamma[g]
        beta = array_beta[-1]
        
        # Computation of the eigenvalues
        eigenValues, eigenVectors = result(u,beta,gamma)
        
        # Loop of the beta values
        for b in range(1,len(array_beta)):  
            beta = array_beta[len(array_beta)-b]  
            
            # Computation of the eigenvalues
            eigenValues, eigenVectors = result(u,beta,gamma)
            
            # Find the most unstable mode
            Arg = np.argmin((-1j*eigenValues).imag)
            
            # Check if the most unstable mode has became unstable
            if (-1j*eigenValues).imag[Arg] < -0.01 and b_critique[g,i] == 0 and (-1j*eigenValues).real[Arg] != 0:
                b_critique[g,i] = beta

# Plotting of the results
plt.xlim((0,1))
plt.ylim((0,24))
plt.plot(b_critique[0,:],u_array,label='gamma = -10',color=(0.3,0.3,0.3),linestyle="dotted")
plt.plot(b_critique[1,:],u_array,label='gamma = 0',color=(0.3,0.3,0.3),linestyle="solid")
plt.plot(b_critique[2,:],u_array,label='gamma = 10',color=(0.3,0.3,0.3),linestyle="dashed")
plt.plot(b_critique[3,:],u_array,label='gamma = 100',color=(0.3,0.3,0.3),linestyle="dashdot")
plt.legend()
plt.xlabel("beta")
plt.ylabel("Adimensional critical flowrate")
plt.title("Flutter instability")
plt.show()
