# =============================================================================
# This code is used to determine the optimal activation function of a NN for 
# classic PINNs
# =============================================================================

# Loading of the librairies
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rd

# Non dimensional parameters of the pipe
d = 6.35*10**-3
D = 15.875*10**-3
M_silicone = 1.34/1000*100**3
m = M_silicone*np.pi*(D**2-d**2)/4
M = 996*np.pi*d**2/4
Estar = 5333.901583749264
E = 225171.9440703812
I = np.pi*(D**4-d**4)/64
L = 46*10**-2
g = 9.81
U = 2
alpha = (I/(E*(M+m)))**0.5*Estar/L**2
beta = M/(M+m)
gamma = (M+m)*L**3*g/(E*I)
u = (M/(E*I))**0.5*L*U
tmax = 1

# Number of beam mode shapes used in the Galerkin decomposition
N = 4

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

# Function used to force the BC on z=0
def f_BC(x):
    return tf.tanh(x)**2

# Function used to initialize the weights with a normal law
def xavier_init(size,name_w):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32, name = name_w)

# Neural network function with sin activation function
def neural_net(X, weights, biases):
    H = X
    num_layers = len(weights) + 1
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.sin(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

# Initialize the weights and biases of the NN
def initialize_NN(layers,name_nn=''):        
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        # Initialize the weights with the xavier function
        W = xavier_init(size=[layers[l], layers[l+1]],name_w = 'weights_'+name_nn+str(l))
        # Initialize the biases at 0
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32, name ='biases_'+name_nn+str(l))
        weights.append(W)
        biases.append(b)     
    return weights, biases

# Deflection of the pipe computed from the PINN model
def NN_time(x,t,weights,biases):
    out_nn = neural_net(tf.transpose(tf.stack([x,t])),weights,biases)[:,0]
    return tf.transpose(out_nn*f_BC(x))

# Derivatives used in the PDE penalization
def y_x(x,t):
    return tf.gradients(NN_time(x,t,W,b),x)[0]
 
def y_xx(x,t):
    return tf.gradients(y_x(x,t),x)[0]

def y_xxx(x,t):
    return tf.gradients(y_xx(x,t),x)[0]

def y_xxxx(x,t):
    return tf.gradients(y_xxx(x,t),x)[0]

def y_t(x,t):
    return tf.gradients(NN_time(x,t,W,b),t)[0]

def y_tt(x,t):
    return tf.gradients(y_t(x,t),t)[0]

def y_xt(x,t):
    return tf.gradients(y_x(x,t),t)[0]

def y_xxxxt(x,t):
    return tf.gradients(y_xxxx(x,t),t)[0]

# Residuals part of the loss
def L_r(x_tf,t_tf):
    residuals = alpha*y_xxxxt(x_tf,t_tf)/u**2 + \
                y_xxxx(x_tf,t_tf)/u**2 + \
                (u**2*ones_Residuals-gamma*(ones_Residuals-x_tf))*y_xx(x_tf,t_tf)/u**2 + \
                2*beta**0.5*y_xt(x_tf,t_tf)/u + \
                gamma*y_x(x_tf,t_tf)/u**2 + \
                y_tt(x_tf,t_tf)/u**2
    return tf.reduce_mean(tf.square(residuals))
           
# Data part of the loss     
def L_d(x_tf,t_tf,w_tf):
    residuals = NN_time(x_tf,t_tf,W,b) - w_tf
    return tf.reduce_mean(tf.square(residuals))

# =============================================================================
# Models trained with sin activation function
# =============================================================================
# Loop to train 10 models and obtain average results
for case in range(1,11):
    print(case)
    N_data = 100
    N_residuals = 100
    N_val = 100
    xData = []
    tData = []
    xData = np.array([rd.random() for i in range(N_data)])
    tData = np.array([rd.random()*tmax for i in range(N_data)])
    wData = np.real(eta(xData,tData))
    xData_tf = tf.constant(xData,dtype=tf.float32,shape=[N_data,])
    tData_tf = tf.constant(tData,dtype=tf.float32,shape=[N_data,])
    wData_tf = tf.constant(wData,dtype=tf.float32,shape=[N_data,])
    tResiduals = np.array([rd.random()*tmax for i in range(N_residuals)])
    xResiduals = np.array([rd.random() for i in range(N_residuals)])
    xResiduals_tf = tf.constant(xResiduals,dtype=tf.float32,shape=[N_residuals,])
    tResiduals_tf = tf.constant(tResiduals,dtype=tf.float32,shape=[N_residuals,])
    xData_validation = []
    tData_validation = []
    xData_validation = np.array([rd.random() for i in range(N_val)])
    tData_validation = np.array([rd.random()*tmax for i in range(N_val)])
    wData_validation = np.real(eta(xData_validation,tData_validation))
    xData_tf_validation = tf.constant(xData_validation,dtype=tf.float32,shape=[N_val,])
    tData_tf_validation = tf.constant(tData_validation,dtype=tf.float32,shape=[N_val,])
    wData_tf_validation = tf.constant(wData_validation,dtype=tf.float32,shape=[N_val,])
    ones_Residuals = tf.ones(N_residuals,dtype=tf.float32)
    layers = [2,20,20,1]
    W,b = initialize_NN(layers)
    wr = 0.01
    loss_1 = wr*L_r(xResiduals_tf,tResiduals_tf) + (1-wr)*L_d(xData_tf,tData_tf,wData_tf) 
    loss_2 = (1-wr)*L_d(xData_tf,tData_tf,wData_tf) 
    loss_v2 = (1-wr)*L_d(xData_tf_validation,tData_tf_validation,wData_tf_validation)
    lr = 1e-5
    optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    optimizer = optimizer_Adam.minimize(loss_1)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False))
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    loss_value_1 = sess.run(loss_1)
    Loss_list_Adam_1 = []
    Loss_list_Adam_2 = []
    V2_Loss_list_Adam = []
    count = []
    it = 0
    itdisp = 100
    tolAdam = 1e-8
    while(loss_value_1>tolAdam and it<=500000):
        sess.run(optimizer)
        loss_value_1 = sess.run(loss_1)
        if it%itdisp == 0:
            Loss_list_Adam_1.append(loss_value_1)
            Loss_list_Adam_2.append(sess.run(loss_2))
            V2_Loss_list_Adam.append(sess.run(loss_v2))
            count.append(it)
            print('Adam it %d - Loss value :  %.3e' % (it, loss_value_1))
        it += 1
    N_plot = 1000
    t_plots = np.linspace(0,tmax,N_plot)
    x_PINN = sess.run(NN_time(tf.constant(L*np.ones(N_plot),dtype=tf.float32,shape=[N_plot,]),tf.constant(t_plots,dtype=tf.float32,shape=[N_plot,]),W,b))
    np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Classic_Sin_Maxiter\TXXNN_Tip_"+str(case)+".txt",np.array([t_plots,eta(L,t_plots),x_PINN]))
    np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Classic_Sin_Maxiter\Training_loss_1_"+str(case)+".txt",np.array([count,Loss_list_Adam_1]))
    np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Classic_Sin_Maxiter\Training_loss_2_"+str(case)+".txt",np.array([count,Loss_list_Adam_2]))
    np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Classic_Sin_Maxiter\Validation_Loss_"+str(case)+".txt",np.array([count,V2_Loss_list_Adam]))

# New definition of the NN with tanh activation functions
def neural_net(X, weights, biases):
    H = X
    num_layers = len(weights) + 1
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

# =============================================================================
# Models trained with tanh activation function
# =============================================================================
# Loop to train 10 models and obtain average results
for case in range(1,11):
    print(case)
    N_data = 100
    N_residuals = 100
    N_val = 100
    xData = []
    tData = []
    xData = np.array([rd.random() for i in range(N_data)])
    tData = np.array([rd.random()*tmax for i in range(N_data)])
    wData = np.real(eta(xData,tData))
    xData_tf = tf.constant(xData,dtype=tf.float32,shape=[N_data,])
    tData_tf = tf.constant(tData,dtype=tf.float32,shape=[N_data,])
    wData_tf = tf.constant(wData,dtype=tf.float32,shape=[N_data,])
    tResiduals = np.array([rd.random()*tmax for i in range(N_residuals)])
    xResiduals = np.array([rd.random() for i in range(N_residuals)])
    xResiduals_tf = tf.constant(xResiduals,dtype=tf.float32,shape=[N_residuals,])
    tResiduals_tf = tf.constant(tResiduals,dtype=tf.float32,shape=[N_residuals,])
    xData_validation = []
    tData_validation = []
    xData_validation = np.array([rd.random() for i in range(N_val)])
    tData_validation = np.array([rd.random()*tmax for i in range(N_val)])
    wData_validation = np.real(eta(xData_validation,tData_validation))
    xData_tf_validation = tf.constant(xData_validation,dtype=tf.float32,shape=[N_val,])
    tData_tf_validation = tf.constant(tData_validation,dtype=tf.float32,shape=[N_val,])
    wData_tf_validation = tf.constant(wData_validation,dtype=tf.float32,shape=[N_val,])
    ones_Residuals = tf.ones(N_residuals,dtype=tf.float32)
    layers = [2,20,20,1]
    W,b = initialize_NN(layers)
    wr = 0.01
    loss_1 = wr*L_r(xResiduals_tf,tResiduals_tf) + (1-wr)*L_d(xData_tf,tData_tf,wData_tf) 
    loss_2 = (1-wr)*L_d(xData_tf,tData_tf,wData_tf) 
    loss_v2 = (1-wr)*L_d(xData_tf_validation,tData_tf_validation,wData_tf_validation)
    lr = 1e-5
    optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    optimizer = optimizer_Adam.minimize(loss_1)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False))
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    loss_value_1 = sess.run(loss_1)
    Loss_list_Adam_1 = []
    Loss_list_Adam_2 = []
    V2_Loss_list_Adam = []
    count = []
    it = 0
    itdisp = 100
    tolAdam = 1e-8
    while(loss_value_1>tolAdam and it<=500000):
        sess.run(optimizer)
        loss_value_1 = sess.run(loss_1)
        if it%itdisp == 0:
            Loss_list_Adam_1.append(loss_value_1)
            Loss_list_Adam_2.append(sess.run(loss_2))
            V2_Loss_list_Adam.append(sess.run(loss_v2))
            count.append(it)
            print('Adam it %d - Loss value :  %.3e' % (it, loss_value_1))
        it += 1
    N_plot = 1000
    t_plots = np.linspace(0,tmax,N_plot)
    x_PINN = sess.run(NN_time(tf.constant(L*np.ones(N_plot),dtype=tf.float32,shape=[N_plot,]),tf.constant(t_plots,dtype=tf.float32,shape=[N_plot,]),W,b))
    np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Classic_Tanh_Maxiter\TXXNN_Tip_"+str(case)+".txt",np.array([t_plots,eta(L,t_plots),x_PINN]))
    np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Classic_Tanh_Maxiter\Training_loss_1_"+str(case)+".txt",np.array([count,Loss_list_Adam_1]))
    np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Classic_Tanh_Maxiter\Training_loss_2_"+str(case)+".txt",np.array([count,Loss_list_Adam_2]))
    np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Classic_Tanh_Maxiter\Validation_Loss_"+str(case)+".txt",np.array([count,V2_Loss_list_Adam]))
    