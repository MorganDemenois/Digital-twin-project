import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rd

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
N = 4
tmax = 1
N_nn = 8

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
M = np.eye(N)
for i in range(N):
    for j in range(N):
        B[i,j] = bsr(i,j)  
        C[i,j] = csr(i,j)
        D[i,j] = dsr(i,j)

Delta = np.zeros((N,N))
FF = np.zeros((N,N))
for i in range(N):
    Delta[i,i] = LAMBDA[i]**4
    FF[i,i] = alpha*LAMBDA[i]**4
    
def result(u):
    S = 2*(beta**0.5)*u*B + FF
    K = Delta + (u**2-gamma)*C + gamma*B + gamma*D
    F = np.block([[np.zeros((N,N)),M],[M,S]])
    E = np.block([[-M,np.zeros((N,N))],[np.zeros((N,N)),K]])     
    eigenValues, eigenVectors = np.linalg.eig(-np.dot(np.linalg.inv(F),E))
    return eigenValues, eigenVectors

eigenValues, eigenVectors = result(u)

CL = np.zeros(2*N)
CL[0] = 0.3
Constant = np.dot(np.linalg.inv(eigenVectors),CL)

def eta(x,t):
    Sol = 0
    for r in range(N):
        qr = 0
        for i in range(2*N):
            qr += Constant[i]*np.exp(eigenValues[i]*t)*eigenVectors[r,i]
        phir = np.cosh(LAMBDA[r]*x)-np.cos(LAMBDA[r]*x)-sigma(r)*(np.sinh(LAMBDA[r]*x)-np.sin(LAMBDA[r]*x))
        Sol += qr*phir
    return Sol

Omega_P = eigenValues

def f_BC(x):
    return tf.tanh(x)**2

def xavier_init(size,name_w):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32, name = name_w)

def neural_net(X, weights, biases):
    H = X
    num_layers = len(weights) + 1
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

def initialize_NN(layers,name_nn=''):        
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W_1 = xavier_init(size=[layers[l], layers[l+1]],name_w = 'weights_'+name_nn+str(l))
        W_2 = xavier_init(size=[layers[l], layers[l+1]],name_w = 'weights_'+name_nn+str(l))
        b_1 = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32, name ='biases_'+name_nn+str(l))
        b_2 = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32, name ='biases_'+name_nn+str(l))
        W = tf.complex(W_1, W_2)
        b = tf.complex(b_1, b_2)
        weights.append(W)
        biases.append(b)     
    return weights, biases

def out_nn_modes(x,weights,biases):
    xint = tf.complex(x,0.)
    out_nn = neural_net(tf.transpose(tf.stack([xint])),weights,biases)[:,:N_nn]
    fbc = tf.complex(f_BC(x),0.)
    t_parts = tf.convert_to_tensor([fbc*out_nn[:,k] for k in range(N_nn)])
    return tf.transpose(t_parts)

def NN_time(x,t,weights,biases,Omega):
    out_NN = out_nn_modes(x,weights,biases)
    parts = [out_NN[:,k]*tf.exp(Omega[k]*tf.complex(t,0.)) for k in range(N_nn)]
    t_parts = tf.convert_to_tensor(parts)
    t_real = tf.real(tf.reduce_sum(t_parts,axis=0))
    return t_real 
           
def L_d(x_tf,t_tf,w_tf):
    residuals = NN_time(x_tf,t_tf,W,b,Omega_P) - w_tf
    return tf.reduce_mean(tf.square(residuals))


for Width in np.array([2,4,6,10,20]):
    for Depth in np.array([1,3,4,5]):
        N_data = 500
        xData = np.array([rd.random() for i in range(N_data)])
        tData = np.array([rd.random()*tmax for i in range(N_data)])
        wData = np.real(eta(xData,tData))
        xData_tf = tf.constant(xData,dtype=tf.float32,shape=[N_data,])
        tData_tf = tf.constant(tData,dtype=tf.float32,shape=[N_data,])
        wData_tf = tf.constant(wData,dtype=tf.float32,shape=[N_data,])
        hidden = Depth*[Width]
        layers = [1]+hidden+[N_nn]
        W,b = initialize_NN(layers)
        loss_1 = L_d(xData_tf,tData_tf,wData_tf) 
        lr = 1e-5
        optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        optimizer = optimizer_Adam.minimize(loss_1)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False))
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        loss_value_1 = sess.run(loss_1)
        Loss_list_Adam_1 = []
        count = []
        it = 0
        itdisp = 100
        tolAdam = 1e-10
        while(loss_value_1>tolAdam and it<=500000):
            sess.run(optimizer)
            loss_value_1 = sess.run(loss_1)
            if it%itdisp == 0:
                Loss_list_Adam_1.append(loss_value_1)
                count.append(it)
                print('Adam it %d - Loss value :  %.3e' % (it, loss_value_1))
            it += 1
        np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Modal_Training_Loss_Depth="+str(Depth)+"_Width="+str(Width)+".txt",np.array([count,Loss_list_Adam_1]))
    

Depth = 2
for Width in np.array([2,4,6,8,12,16,20,26,34]):
    N_data = 500
    xData = np.array([rd.random() for i in range(N_data)])
    tData = np.array([rd.random()*tmax for i in range(N_data)])
    wData = np.real(eta(xData,tData))
    xData_tf = tf.constant(xData,dtype=tf.float32,shape=[N_data,])
    tData_tf = tf.constant(tData,dtype=tf.float32,shape=[N_data,])
    wData_tf = tf.constant(wData,dtype=tf.float32,shape=[N_data,])
    hidden = Depth*[Width]
    layers = [1]+hidden+[N_nn]
    W,b = initialize_NN(layers)
    loss_1 = L_d(xData_tf,tData_tf,wData_tf) 
    lr = 1e-5
    optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    optimizer = optimizer_Adam.minimize(loss_1)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False))
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    loss_value_1 = sess.run(loss_1)
    Loss_list_Adam_1 = []
    count = []
    it = 0
    itdisp = 100
    tolAdam = 1e-10
    while(loss_value_1>tolAdam and it<=500000):
        sess.run(optimizer)
        loss_value_1 = sess.run(loss_1)
        if it%itdisp == 0:
            Loss_list_Adam_1.append(loss_value_1)
            count.append(it)
            print('Adam it %d - Loss value :  %.3e' % (it, loss_value_1))
        it += 1
    np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Modal_Training_Loss_Depth="+str(Depth)+"_Width="+str(Width)+".txt",np.array([count,Loss_list_Adam_1]))

Depth = 2
for Width in np.array([2,4,6,8,12,16,20]):
    count,Loss_list_Adam_1 = np.loadtxt("C:\\Users\Morgan\Videos\PINN_theory\Modal_Size\Modal_Training_Loss_Depth="+str(Depth)+"_Width="+str(Width)+".txt")
    plt.plot(count,Loss_list_Adam_1,label=Width)
plt.yscale("log")    
plt.legend()
plt.show()

Width = 6
for Depth in np.array([1,2,3,4,5]):
    count,Loss_list_Adam_1 = np.loadtxt("C:\\Users\Morgan\Videos\PINN_theory\Modal_Size\Modal_Training_Loss_Depth="+str(Depth)+"_Width="+str(Width)+".txt")
    plt.plot(count,Loss_list_Adam_1,label=Depth)
plt.yscale("log")    
plt.legend()
plt.show()


# Plotting of the results
styles = ["solid","dashed","dotted","solid","dashed","dotted"]
colors = [(0.6,0.6,0.6),(0.6,0.6,0.6),(0.6,0.6,0.6),(0.,0.,0.),(0.,0.,0.),(0.,0.,0.)]
Width = 6
j = 0
for Depth in np.array([1,2,3,4,5]):
    count,Loss_list_Adam_1 = np.loadtxt("C:\\Users\Morgan\Videos\PINN_theory\Modal_Size\Modal_Training_Loss_Depth="+str(Depth)+"_Width="+str(Width)+".txt")
    reduced_count = []
    reduced_loss = []
    for i in range(500-10):
        imin = 10*i+np.argmin(Loss_list_Adam_1[10*i:10*(i+1)])
        reduced_count.append(count[imin])
        reduced_loss.append(Loss_list_Adam_1[imin])
    plt.plot(reduced_count,reduced_loss,label=Depth,color=colors[j],linestyle=styles[j])
    plt.yscale("log")
    j += 1
plt.legend()
plt.xlabel("Iteration number")
plt.ylabel("Training loss")
plt.show()

Depth = 2
j = 0
for Width in np.array([4,6,8,12,16,20]):
    count,Loss_list_Adam_1 = np.loadtxt("C:\\Users\Morgan\Videos\PINN_theory\Modal_Size\Modal_Training_Loss_Depth="+str(Depth)+"_Width="+str(Width)+".txt")
    reduced_count = []
    reduced_loss = []
    for i in range(500-10):
        imin = 10*i+np.argmin(Loss_list_Adam_1[10*i:10*(i+1)])
        reduced_count.append(count[imin])
        reduced_loss.append(Loss_list_Adam_1[imin])
    plt.plot(reduced_count,reduced_loss,label=Width,color=colors[j],linestyle=styles[j])
    plt.yscale("log")
    j += 1
plt.legend()
plt.xlabel("Iteration number")
plt.ylabel("Training loss")
plt.show()