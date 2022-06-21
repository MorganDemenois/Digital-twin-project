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
U = 4
alpha = (I/(E*(M+m)))**0.5*Estar/L**2
beta = M/(M+m)
gamma = (M+m)*L**3*g/(E*I)
u = (M/(E*I))**0.5*L*U
N = 10

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

tmax = 1
N_data = 5000
N_residuals = 500

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
xData_validation = np.array([rd.random() for i in range(N_data)])
tData_validation = np.array([rd.random()*tmax for i in range(N_data)])
wData_validation = np.real(eta(xData_validation,tData_validation))

xData_tf_validation = tf.constant(xData_validation,dtype=tf.float32,shape=[N_data,])
tData_tf_validation = tf.constant(tData_validation,dtype=tf.float32,shape=[N_data,])
wData_tf_validation = tf.constant(wData_validation,dtype=tf.float32,shape=[N_data,])

tResiduals_validation = np.array([rd.random()*tmax for i in range(N_residuals)])
xResiduals_validation = np.array([rd.random() for i in range(N_residuals)])
xResiduals_tf_validation = tf.constant(xResiduals_validation,dtype=tf.float32,shape=[N_residuals,])
tResiduals_tf_validation = tf.constant(tResiduals_validation,dtype=tf.float32,shape=[N_residuals,])

ones_Residuals = tf.ones(N_residuals,dtype=tf.float32)

u_guess = 5
u_tf = tf.abs(tf.Variable(u_guess,dtype=tf.float32,name="u_opt"))

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
        H = tf.sin(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

def initialize_NN(layers,name_nn=''):        
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]],name_w = 'weights_'+name_nn+str(l))
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32, name ='biases_'+name_nn+str(l))
        weights.append(W)
        biases.append(b)     
    return weights, biases

layers = [2,20,20,1]
W,b = initialize_NN(layers)

def NN_time(x,t,weights,biases):
    out_nn = neural_net(tf.transpose(tf.stack([x,t])),weights,biases)[:,0]
    return tf.transpose(out_nn*f_BC(x))

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

def L_r(x_tf,t_tf):
    residuals = alpha*y_xxxxt(x_tf,t_tf)/u_tf**2 + \
                y_xxxx(x_tf,t_tf)/u_tf**2 + \
                (u_tf**2*ones_Residuals-gamma*(ones_Residuals-x_tf))*y_xx(x_tf,t_tf)/u_tf**2 + \
                2*beta**0.5*y_xt(x_tf,t_tf)/u_tf + \
                gamma*y_x(x_tf,t_tf)/u_tf**2 + \
                y_tt(x_tf,t_tf)/u_tf**2
    return tf.reduce_mean(tf.square(residuals))
           
def L_d(x_tf,t_tf,w_tf):
    residuals = NN_time(x_tf,t_tf,W,b) - w_tf
    return tf.reduce_mean(tf.square(residuals))


wr = 0.01
loss_1 = wr*L_r(xResiduals_tf,tResiduals_tf) + (1-wr)*L_d(xData_tf,tData_tf,wData_tf) 
loss_2 = (1-wr)*L_d(xData_tf,tData_tf,wData_tf) 
loss_v1 = wr*L_r(xResiduals_tf_validation,tResiduals_tf_validation) + (1-wr)*L_d(xData_tf_validation,tData_tf_validation,wData_tf_validation) 
loss_v2 = (1-wr)*L_d(xData_tf_validation,tData_tf_validation,wData_tf_validation)
# loss_1 = (1-wr)*L_d(xData_tf,tData_tf,wData_tf) 
# loss_2 = (1-wr)*L_d(xData_tf,tData_tf,wData_tf) 
# loss_v1 = (1-wr)*L_d(xData_tf_validation,tData_tf_validation,wData_tf_validation) 
# loss_v2 = (1-wr)*L_d(xData_tf_validation,tData_tf_validation,wData_tf_validation)

lr = 1e-5
optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
optimizer = optimizer_Adam.minimize(loss_1)

optimizer_LBFGSB = tf.contrib.opt.ScipyOptimizerInterface(loss_1, method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000, 
                                                                            'maxfun': 50000, 
                                                                            'maxcor': 50,
                                                                            'maxls': 50,
                                                                            'ftol' : 1.0 * np.finfo(float).eps}) 
Param_list_LF = []
Loss_list_1_LF = []
Loss_list_2_LF = []
V1_Loss_list_LF = []
V2_Loss_list_LF = []
def print_loss(loss_evaled_1,loss_evaled_2,vector_evaled,val_1,val_2):
    Param_list_LF.append(vector_evaled)
    Loss_list_1_LF.append(loss_evaled_1)
    Loss_list_2_LF.append(loss_evaled_2)
    V1_Loss_list_LF.append(val_1)
    V2_Loss_list_LF.append(val_2)
    print("Loss = "+str('%.3e' % (loss_evaled_1)),"Param = "+str(vector_evaled))

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False))
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

optimizer_LBFGSB.minimize(sess,
                fetches = [loss_1,loss_2,u_tf,loss_v1,loss_v2],
                loss_callback = print_loss)

plt.style.use('dark_background')
plt.plot(np.log10(np.array(Loss_list_1_LF)))
plt.title("Loss value evolution with Newton")
plt.xlabel("Iteration number")
plt.ylabel("Loss value")
plt.show()
plt.plot(Param_list_LF)
plt.title("Param value evolution with Newton")
plt.xlabel("Iteration number")
plt.ylabel("Param value")
plt.show()
plt.plot(np.log10(np.array(V1_Loss_list_LF)))
plt.title("Validation loss value 1 evolution with Newton")
plt.xlabel("Iteration number")
plt.ylabel("Loss value")
plt.show()
plt.plot(np.log10(np.array(V2_Loss_list_LF)))
plt.title("Validation loss 2 value evolution with Newton")
plt.xlabel("Iteration number")
plt.ylabel("Loss value")
plt.show()

loss_value_1 = sess.run(loss_1)
Loss_list_Adam_1 = []
Loss_list_Adam_2 = []
V1_Loss_list_Adam = []
V2_Loss_list_Adam = []
Param_list_Adam = []
count = []
it = 0
itdisp = 100
tolAdam = 1e-8
while(loss_value_1>tolAdam and it<=900000):
    sess.run(optimizer)
    loss_value_1 = sess.run(loss_1)
    if it%itdisp == 0:
        Param_value = sess.run(u_tf)
        Loss_list_Adam_1.append(loss_value_1)
        Loss_list_Adam_2.append(sess.run(loss_2))
        Param_list_Adam.append(Param_value)
        V1_Loss_list_Adam.append(sess.run(loss_v1))
        V2_Loss_list_Adam.append(sess.run(loss_v2))
        count.append(it)
        print('Adam it %d - Loss value :  %.3e' % (it, loss_value_1))
        print('Adam it %d - Param value : %.3e' % (it, Param_value))
    it += 1

plt.plot(count,np.log10(Loss_list_Adam_1))
plt.title("Loss evolution with Adam")
plt.xlabel("Iteration number")
plt.ylabel("Loss value")
plt.show()

plt.plot(count,Param_list_Adam)
plt.title("Param value evolution with Adam")
plt.xlabel("Iteration number")
plt.ylabel("Param value")
plt.show()

plt.plot(count,np.log10(V1_Loss_list_Adam))
plt.title("Loss evolution with Adam")
plt.xlabel("Iteration number")
plt.ylabel("V1 Loss value")
plt.show()

plt.plot(count,np.log10(Loss_list_Adam_2))
plt.plot(count,np.log10(V2_Loss_list_Adam))
plt.title("Loss evolution with Adam")
plt.xlabel("Iteration number")
plt.ylabel("V2 Loss value")
plt.show()

N_plot = 1000
t_plots = np.linspace(0,tmax,N_plot)
x_PINN = sess.run(NN_time(tf.constant(L*np.ones(N_plot),dtype=tf.float32,shape=[N_plot,]),tf.constant(t_plots,dtype=tf.float32,shape=[N_plot,]),W,b))
plt.plot(t_plots,eta(L,t_plots),label="Pipe position along time from the theory")
plt.plot(t_plots,x_PINN,label="Pipe position along time from the PINN")
plt.title("Comparison between PINN and the videos")
plt.xlabel("Time in seconds")
plt.ylabel("Deflection in meters")
plt.legend()
plt.show()


count_LF = [i for i in range(0,len(Param_list_LF))]
np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\TXXNN_Tip_U=4.txt",np.array([t_plots,eta(L,t_plots),x_PINN]))
np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\FlowOpt_Opt_U=4_Adam.txt",np.array([count,Param_list_Adam]))
np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Training_loss_U=4_Adam_1.txt",np.array([count,Loss_list_Adam_1]))
np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Training_loss_U=4_Adam_2.txt",np.array([count,Loss_list_Adam_2]))
np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Validation_Loss_1_U=4_Adam.txt",np.array([count,V1_Loss_list_Adam]))
np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Validation_Loss_2_U=4_Adam.txt",np.array([count,V2_Loss_list_Adam]))
np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\FlowOpt_Opt_U=4_Newton.txt",np.array([count_LF,Param_list_LF]))
np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Training_loss_1_U=4_Newton.txt",np.array([count_LF,Loss_list_1_LF]))
np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Training_loss_2_U=4_Newton.txt",np.array([count_LF,Loss_list_2_LF]))
np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Validation_Loss_1_U=4_Newton.txt",np.array([count_LF,V1_Loss_list_LF]))
np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\Validation_Loss_2_U=4_Newton.txt",np.array([count_LF,V2_Loss_list_LF]))
  

W_array = sess.run(W)
b_array = sess.run(b)
for i in range(len(layers)-1):
    np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\W_U=4_"+str(i)+".txt",W_array[i])
    np.savetxt("C:\\Users\Morgan\Videos\PINN_theory\B_U=4_"+str(i)+".txt",b_array[i])
    
# W_array_test = np.empty(3,dtype=object)
# b_array_test = np.empty(3,dtype=object)

# for i in range(0,len(layers)-1):
#     W_array_test[i] = np.loadtxt("C:\\Users\Morgan\Videos\PINN_theory\W_u=14_"+str(i)+".txt")
#     b_array_test[i] = np.loadtxt("C:\\Users\Morgan\Videos\PINN_theory\B_u=14_"+str(i)+".txt")

# W_new,b_new = restore_one_NN([2,20,20,1], W_array_test, b_array_test)
# sess.run(NN_time(tf.constant(0.4,dtype=tf.float32,shape=(1,)),tf.constant(1.2,dtype=tf.float32,shape=(1,)),W,b))
# sess.run(NN_time(tf.constant(0.4,dtype=tf.float32,shape=(1,)),tf.constant(1.2,dtype=tf.float32,shape=(1,)),W_new,b_new))

