import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rd
import tikzplotlib as tpl
plt.rcParams.update({"text.usetex": True,"font.family": "serif","font.serif": ["Computer Modern Roman"]})
plt.rcParams['figure.autolayout'] = True


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

def y_x(x,t):
    return tf.gradients(NN_time(x,t,W,b,Omega_P),x)[0]
 
def y_xx(x,t):
    return tf.gradients(y_x(x,t),x)[0]

def y_xxx(x,t):
    return tf.gradients(y_xx(x,t),x)[0]

def y_xxxx(x,t):
    return tf.gradients(y_xxx(x,t),x)[0]

def y_t(x,t):
    return tf.gradients(NN_time(x,t,W,b,Omega_P),t)[0]

def y_tt(x,t):
    return tf.gradients(y_t(x,t),t)[0]

def y_xt(x,t):
    return tf.gradients(y_x(x,t),t)[0]

def y_xxxxt(x,t):
    return tf.gradients(y_xxxx(x,t),t)[0]

# def L_r(x_tf,t_tf):
#     residuals = alpha*y_xxxxt(x_tf,t_tf)/u**2 + \
#                 y_xxxx(x_tf,t_tf)/u**2 + \
#                 (u**2*ones_Residuals-gamma*(ones_Residuals-x_tf))*y_xx(x_tf,t_tf)/u**2 + \
#                 2*beta**0.5*y_xt(x_tf,t_tf)/u + \
#                 gamma*y_x(x_tf,t_tf)/u**2 + \
#                 y_tt(x_tf,t_tf)/u**2
#     return tf.reduce_mean(tf.square(residuals))
           
def L_d(x_tf,t_tf,w_tf):
    residuals = NN_time(x_tf,t_tf,W,b,Omega_P) - w_tf
    return tf.reduce_mean(tf.square(residuals))

for N_data in np.array([10,20,40,60,80,100,150,200,250,300,400,500]):
    # for N_data in np.array([25]):
        # for case in range(8,10):
    N_val = 1000
    xData = []
    tData = []
    xData = np.array([rd.random() for i in range(N_data)])
    tData = np.array([rd.random()*tmax for i in range(N_data)])
    wData = np.real(eta(xData,tData))
    xData_tf = tf.constant(xData,dtype=tf.float32,shape=[N_data,])
    tData_tf = tf.constant(tData,dtype=tf.float32,shape=[N_data,])
    wData_tf = tf.constant(wData,dtype=tf.float32,shape=[N_data,])
    # tResiduals = np.array([rd.random()*tmax for i in range(N_residuals)])
    # xResiduals = np.array([rd.random() for i in range(N_residuals)])
    # xResiduals_tf = tf.constant(xResiduals,dtype=tf.float32,shape=[N_residuals,])
    # tResiduals_tf = tf.constant(tResiduals,dtype=tf.float32,shape=[N_residuals,])
    xData_validation = []
    tData_validation = []
    xData_validation = np.array([rd.random() for i in range(N_val)])
    tData_validation = np.array([rd.random()*tmax for i in range(N_val)])
    wData_validation = np.real(eta(xData_validation,tData_validation))
    xData_tf_validation = tf.constant(xData_validation,dtype=tf.float32,shape=[N_val,])
    tData_tf_validation = tf.constant(tData_validation,dtype=tf.float32,shape=[N_val,])
    wData_tf_validation = tf.constant(wData_validation,dtype=tf.float32,shape=[N_val,])
    # ones_Residuals = tf.ones(N_residuals,dtype=tf.float32)
    layers = [1,20,20,N_nn]
    W,b = initialize_NN(layers)
    wr = 0.0
    # loss_1 = wr*L_r(xResiduals_tf,tResiduals_tf) + (1-wr)*L_d(xData_tf,tData_tf,wData_tf) 
    loss_1 = (1-wr)*L_d(xData_tf,tData_tf,wData_tf) 
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
    tolAdam = 2e-5
    while(loss_value_1>tolAdam and it<=1000000):
        sess.run(optimizer)
        loss_value_1 = sess.run(loss_1)
        if it%itdisp == 0:
            Loss_list_Adam_1.append(loss_value_1)
            Loss_list_Adam_2.append(sess.run(loss_2))
            V2_Loss_list_Adam.append(sess.run(loss_v2))
            count.append(it)
            print( 'Adam it %d - Loss value :  %.3e' % (it, loss_value_1))
        it += 1
    N_plot = 1000
    t_plots = np.linspace(0,tmax,N_plot)
    x_PINN = sess.run(NN_time(tf.constant(L*np.ones(N_plot),dtype=tf.float32,shape=[N_plot,]),tf.constant(t_plots,dtype=tf.float32,shape=[N_plot,]),W,b,Omega_P))
    plt.plot(t_plots,x_PINN)
    plt.plot(t_plots,eta(L,t_plots))
    plt.show()
    np.savetxt(r"C:\\Users\Morgan\Videos\PINN_Theory\Modal_PINN_Data_Impact_Ndata=50_Error=1e-5\Validation_Loss_1_"+str(N_data)+".txt",np.array([count,V2_Loss_list_Adam]))
    np.savetxt(r"C:\\Users\Morgan\Videos\PINN_Theory\Modal_PINN_Data_Impact_Ndata=50_Error=1e-5\Training_Loss_1_1_"+str(N_data)+".txt",np.array([count,Loss_list_Adam_1]))
    np.savetxt(r"C:\\Users\Morgan\Videos\PINN_Theory\Modal_PINN_Data_Impact_Ndata=50_Error=1e-5\Training_Loss_2_1_"+str(N_data)+".txt",np.array([count,Loss_list_Adam_2]))
# count_array = np.zeros((9,9))
# val_array = np.zeros((9,9))
# val_list = []
# count_list = []
# Residuals_list = [1,5,10,20,40,80,150,300]
# case_list = [6,6,6,9,9,8,6,2]
# index = [3,4,0,-1,5,4,5,1] 
# index_count = [2,0,2,2,4,6,1,0] 
# for i in range(8):
#     N_residuals = Residuals_list[i]
#     for case in range(1,case_list[i]+1):
#         count,val = np.loadtxt(r"C:\\Users\Morgan\Videos\PINN_Theory\Modal_PINN_Residuals_Impact_Ndata=50_Error=1e-5\Validation_Loss_Nr="+str(N_residuals)+"_Case="+str(case)+".txt")
#         val_array[case-1,i] = val[-1]
#         count_array[case-1,i] = count[-1]

# for i in range(len(index)):
#     val_list.append(val_array[index[i],i])
#     count_list.append(count_array[index_count[i],i])
    


# plt.plot([1,5,10,20,40,80,150,300],val_list,label="val")
# plt.yscale('log')
# plt.legend()
# plt.show()

# plt.plot([1,5,10,20,40,80,150,300],count_list,label="val")
# plt.legend()
# plt.show()


# count_array = np.zeros((6,9))
# val_array = np.zeros((6,9))
# val_list_classic = []
# count_list_classic = []
# Residuals_list = [1,5,10,20,40,80,150,300]
# case_list = [3,3,3,3,3,3,3,3]
# index = [2,0,0,2,0,2,0,0] 
# for i in range(8):
#     N_residuals = Residuals_list[i]
#     for case in range(1,case_list[i]+1):
#         count,val = np.loadtxt(r"C:\\Users\Morgan\Videos\PINN_Theory\Classic_PINN_Residuals_Impact_Ndata=50_Error=1e-5\Validation_Loss_Nr="+str(N_residuals)+"_Case="+str(case)+".txt")
#         val_array[case-1,i] = val[-1]
#         count_array[case-1,i] = count[-1]

# for i in range(len(index)):
#     val_list_classic.append(val_array[index[i],i])
#     count_list_classic.append(count_array[index[i],i])


# plt.plot([1,5,10,20,40,80,150,300],val_list_classic,color=(0.3,0.3,0.3),label="classic")
# plt.plot([1,5,10,20,40,80,150,300],val_list,color=(0.3,0.3,0.3),label="modal",linestyle="dashed")
# plt.ylabel("Test loss")
# plt.xlabel("Number of penalization points")
# plt.yscale('log')
# plt.legend()
# tpl.save(r"C:\Users\Morgan\Desktop\Loss_Nresiduals.tex")
# plt.show()

# plt.plot([1,5,10,20,40,80,150,300],count_list_classic,color=(0.3,0.3,0.3),label="classic")
# plt.plot([1,5,10,20,40,80,150,300],count_list,color=(0.3,0.3,0.3),label="modal",linestyle="dashed")
# plt.ylabel("Number of iteration")
# plt.xlabel("Number of penalization points")
# plt.legend()
# tpl.save(r"C:\Users\Morgan\Desktop\Iteration_Nresiduals.tex")
# plt.show()

count_list_classic = []
val_list_classic = []
for N_data in [10,20,40,60,80,100,150,200,250,300,400,500]:
    count,val = np.loadtxt(r"C:\\Users\Morgan\Videos\PINN_Theory\Classic_PINN_Data_Impact_Ndata=50_Error=1e-5\Validation_Loss_2_"+str(N_data)+".txt")
    count_list_classic.append(count[-1])
    val_list_classic.append(val[-1])
    

count_list = []
val_list = []
for N_data in [10,20,40,60,80,100,150,200,250,300,400,500]:
    count,val = np.loadtxt(r"C:\\Users\Morgan\Videos\PINN_Theory\Modal_PINN_Data_Impact_Ndata=50_Error=1e-5\Validation_Loss_2_"+str(N_data)+".txt")
    count_list.append(count[-1])
    val_list.append(val[-1])
    
# plt.plot([10,20,40,60,80,100,150,200,250,300,400,500],val_list,color=(0.3,0.3,0.3),label="modal",linestyle="dashed")
# plt.plot([10,20,40,60,80,100,150,200,250,300,400,500],val_list_classic,color=(0.3,0.3,0.3),label="classic")
# plt.legend()
# plt.yscale("log")
# plt.ylabel("Test loss")
# plt.xlabel("Ndata")
# tpl.save(r"C:\Users\Morgan\Desktop\TestLoss_NData.tex")
# plt.show()

plt.plot([10,20,40,60,80,100,150,200,250,300,400,500],count_list,color=(0.3,0.3,0.3),label="modal",linestyle="dashed")
plt.plot([10,20,40,60,80,100,150,200,250,300,400,500],count_list_classic,color=(0.3,0.3,0.3),label="classic")
plt.legend()
plt.ylabel("Iteration number")
plt.xlabel("Ndata")
tpl.save(r"C:\Users\Morgan\Desktop\Iteration_NData.tex")
plt.show()