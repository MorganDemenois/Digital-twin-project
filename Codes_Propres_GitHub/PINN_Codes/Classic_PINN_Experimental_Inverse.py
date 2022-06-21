import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rd
# import time
# CUDA_VISIBLE_DEVICES=""
U_array = np.loadtxt(r"C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\U_60Hz.txt")

Uv = np.mean(U_array[:150])
# print(Uv)
d = 6.35*10**-3
U = Uv*4/(60*1000*np.pi*d**2)
print(U)
M_silicone = 1.34/1000*100**3
D = 15.875*10**-3
m = M_silicone*np.pi*(D**2-d**2)/4
M = 996*np.pi*d**2/4
I = np.pi*(D**4-d**4)/64
L = 46*10**-2
g = 9.81
Estar = 5333.901583749264
E = 225171.9440703812

alpha = (I/(E*(M+m)))**0.5*Estar/L**2
beta = M/(M+m)
gamma = (M+m)*L**3*g/(E*I)
u = (M/(E*I))**0.5*L*U

N_residuals = 500
Data_training = np.loadtxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\Training_Data_5000Points_PINN_18Hz_1.txt")
Data_validation = np.loadtxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\Validation_Data_5000Points_PINN_18Hz_1.txt")

N_data = Data_training.shape[1]
# xData = Data[1]/L
# tData = Data[2]*(E*I/(M+m))**0.5/L**2
# wData = Data[0]/L
# print(N_data)
xData_training = Data_training[1]
tData_training = Data_training[2]
wData_training = Data_training[0]

xData_training_tf = tf.constant(xData_training,dtype=tf.float32,shape=[N_data,])
tData_training_tf = tf.constant(tData_training,dtype=tf.float32,shape=[N_data,])
wData_training_tf = tf.constant(wData_training,dtype=tf.float32,shape=[N_data,])

xData_validation = Data_validation[1]
tData_validation = Data_validation[2]
wData_validation = Data_validation[0]

xData_validation_tf = tf.constant(xData_validation,dtype=tf.float32,shape=[N_data,])
tData_validation_tf = tf.constant(tData_validation,dtype=tf.float32,shape=[N_data,])
wData_validation_tf = tf.constant(wData_validation,dtype=tf.float32,shape=[N_data,])

tmax = max(tData_training)
# print(tmax)
taumax = (E*I/(M+m))**0.5*tmax/L**2
tResiduals_training = np.array([rd.random()*tmax for i in range(N_residuals)])
xResiduals_training = np.array([rd.random()*L for i in range(N_residuals)])
xResiduals_training_tf = tf.constant(xResiduals_training,dtype=tf.float32,shape=[N_residuals,])
tResiduals_training_tf = tf.constant(tResiduals_training,dtype=tf.float32,shape=[N_residuals,])

tResiduals_validation = np.array([rd.random()*tmax for i in range(N_residuals)])
xResiduals_validation = np.array([rd.random()*L for i in range(N_residuals)])
xResiduals_validation_tf = tf.constant(xResiduals_validation,dtype=tf.float32,shape=[N_residuals,])
tResiduals_validation_tf = tf.constant(tResiduals_validation,dtype=tf.float32,shape=[N_residuals,])

ones_Residuals = tf.ones(N_residuals,dtype=tf.float32)

U_guess = 1
U_tf = tf.Variable(U_guess,dtype=tf.float32,name="U",constraint=lambda t: tf.clip_by_value(t, 0, 10))


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

def restore_one_NN(layers,w_value,b_value):
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = tf.constant(w_value[l],dtype=tf.float32,shape=[layers[l],layers[l+1]])
        b = tf.constant(b_value[l],dtype=tf.float32,shape=[1,layers[l+1]])
        weights.append(W)
        biases.append(b)
    return weights,biases

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

def L_r(xResiduals_tf,tResiduals_tf):

    residuals = Estar*I*y_xxxxt(xResiduals_tf,tResiduals_tf) + \
                E*I*y_xxxx(xResiduals_tf,tResiduals_tf) + \
                (M*U_tf**2*ones_Residuals-(M+m)*g*(L*ones_Residuals-xResiduals_tf))*y_xx(xResiduals_tf,tResiduals_tf) + \
                2*U_tf*M*y_xt(xResiduals_tf,tResiduals_tf) + \
                (M+m)*g*y_x(xResiduals_tf,tResiduals_tf) + \
                (m+M)*y_tt(xResiduals_tf,tResiduals_tf)
    return tf.reduce_mean(tf.square(residuals))

           
def L_d(xData_tf,tData_tf,wData_tf):
    residuals = NN_time(xData_tf,tData_tf,W,b) - wData_tf
    return tf.reduce_mean(tf.square(residuals))


wr = 0.
loss_training_1 = (1-wr)*L_d(xData_training_tf,tData_training_tf,wData_training_tf) + wr*L_r(xResiduals_training_tf,tResiduals_training_tf)
loss_training_2 = (1-wr)*L_d(xData_training_tf,tData_training_tf,wData_training_tf)
loss_validation_1 = (1-wr)*L_d(xData_validation_tf,tData_validation_tf,wData_validation_tf) + wr*L_r(xResiduals_validation_tf,tResiduals_validation_tf)
loss_validation_2 = (1-wr)*L_d(xData_validation_tf,tData_validation_tf,wData_validation_tf)
# loss_validation = L_d(xData_validation_tf,tData_validation_tf,wData_validation_tf) 
# loss_training = L_d(xData_training_tf,tData_training_tf,wData_training_tf) 


lr = 1e-5
optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
optimizer = optimizer_Adam.minimize(loss_training_1)

optimizer_LBFGSB = tf.contrib.opt.ScipyOptimizerInterface(loss_training_1, method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000, 
                                                                            'maxfun': 50000, 
                                                                            'maxcor': 50,
                                                                            'maxls': 50,
                                                                            'ftol' : 1.0 * np.finfo(float).eps}) 
U_list_LF = []
Loss_list_1_LF = []
Loss_list_2_LF = []
Validation_loss_list_LF_1 = []
Validation_loss_list_LF_2 = []
def print_loss(loss_evaled_1, loss_evaled_2, vector_evaled, validation_loss_1,validation_loss_2):
    U_list_LF.append(vector_evaled)
    Loss_list_1_LF.append(loss_evaled_1)
    Loss_list_2_LF.append(loss_evaled_2)
    Validation_loss_list_LF_1.append(validation_loss_1)
    Validation_loss_list_LF_2.append(validation_loss_2)
    print("Loss = "+str('%.3e' % (loss_evaled_1)),"U = "+str(vector_evaled))
    
# def print_loss(loss_evaled):
    # Gamma_list_LF.append(vector_evaled)
    # Loss_list_LF.append(loss_evaled)
    # print("Loss = "+str('%.3e' % (loss_evaled)))

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False))
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

optimizer_LBFGSB.minimize(sess,
                fetches = [loss_training_1,loss_training_2,U_tf,loss_validation_1,loss_validation_2],
                loss_callback = print_loss)

# optimizer_LBFGSB.minimize(sess,
                # fetches = [loss],
                # loss_callback = print_loss)

plt.style.use('dark_background')
plt.plot(np.log10(np.array(Loss_list_1_LF)))
plt.title("Loss value evolution with Newton")
plt.xlabel("Iteration number")
plt.ylabel("Loss value")
plt.show()
plt.style.use('dark_background')
plt.plot(np.log10(np.array(Validation_loss_list_LF_1)))
plt.title("Validation loss value evolution with Newton")
plt.xlabel("Iteration number")
plt.ylabel("Validation loss value")
plt.show()
plt.style.use('dark_background')
plt.plot(np.log10(np.array(Validation_loss_list_LF_2)))
plt.title("Validation loss value evolution with Newton")
plt.xlabel("Iteration number")
plt.ylabel("Validation loss value")
plt.show()
plt.plot(U_list_LF)
plt.title("U value evolution with Newton")
plt.xlabel("Iteration number")
plt.ylabel("U value")
plt.show()

loss_value_1 = sess.run(loss_training_1)
Loss_list_Adam_1 = []
Loss_list_Adam_2 = []
Validation_loss_list_Adam_1 = []
Validation_loss_list_Adam_2 = []
U_list_Adam = []
count = []
it = 0
itdisp = 100
tolAdam = 5e-6
while(loss_value_1>tolAdam and it<=500000):
    sess.run(optimizer)
    loss_value_1 = sess.run(loss_training_1)
    if it%itdisp == 0:
        U_value = sess.run(U_tf)
        Loss_list_Adam_1.append(loss_value_1)
        Loss_list_Adam_2.append(sess.run(loss_training_2))
        U_list_Adam.append(U_value)
        Validation_loss_list_Adam_1.append(sess.run(loss_validation_1))
        Validation_loss_list_Adam_2.append(sess.run(loss_validation_2))
        count.append(it)
        print('Adam it %d - Loss value :  %.3e' % (it, loss_value_1))
        print('Adam it %d - U value : %.3e' % (it, U_value))
    it += 1

wr = 0.05
loss_training_1 = (1-wr)*L_d(xData_training_tf,tData_training_tf,wData_training_tf) + wr*L_r(xResiduals_training_tf,tResiduals_training_tf)
loss_training_2 = (1-wr)*L_d(xData_training_tf,tData_training_tf,wData_training_tf)
loss_validation_1 = (1-wr)*L_d(xData_validation_tf,tData_validation_tf,wData_validation_tf) + wr*L_r(xResiduals_validation_tf,tResiduals_validation_tf)
loss_validation_2 = (1-wr)*L_d(xData_validation_tf,tData_validation_tf,wData_validation_tf)
optimizer = optimizer_Adam.minimize(loss_training_1)
loss_value_1 = sess.run(loss_training_1)
# it_start = it
# start = time.time()
while(loss_value_1>tolAdam and it<=500000):
    sess.run(optimizer)
    loss_value_1 = sess.run(loss_training_1)
    # if it-it_start>10000:
        # print(time.time()-start)
    if it%itdisp == 0:
        U_value = sess.run(U_tf)
        Loss_list_Adam_1.append(loss_value_1)
        Loss_list_Adam_2.append(sess.run(loss_training_2))
        U_list_Adam.append(U_value)
        Validation_loss_list_Adam_1.append(sess.run(loss_validation_1))
        Validation_loss_list_Adam_2.append(sess.run(loss_validation_2))
        count.append(it)
        print('Adam it %d - Loss value :  %.3e' % (it, loss_value_1))
        print('Adam it %d - U value : %.3e' % (it, U_value))
    it += 1
     
plt.plot(count,np.log10(Validation_loss_list_Adam_1))
plt.title("Validation loss evolution with Adam")
plt.xlabel("Iteration number")
plt.ylabel("Validation loss value")
plt.show()

plt.plot(count,np.log10(Validation_loss_list_Adam_2))
plt.title("Validation loss evolution with Adam")
plt.xlabel("Iteration number")
plt.ylabel("Validation loss value")
plt.show()

plt.plot(count,np.log10(Loss_list_Adam_1))
plt.title("Loss evolution with Adam")
plt.xlabel("Iteration number")
plt.ylabel("Loss value")
plt.show()

plt.plot(count,np.log10(Loss_list_Adam_2))
plt.title("Loss evolution with Adam")
plt.xlabel("Iteration number")
plt.ylabel("Loss value")
plt.show()

plt.plot(count,U_list_Adam)
plt.plot([count[0],count[-1]],[U,U])
plt.title("U value evolution with Adam")
plt.xlabel("Iteration number")
plt.ylabel("U value")
plt.show()

# t_array = np.linspace(0,tmax,501)
# x_array = np.linspace(0,L,101)
# x_tf = tf.constant(x_array,dtype=tf.float32,shape=[101])
# erreur = 0
# for t in t_array:  
#     t_tf = tf.constant(t*np.ones(101),dtype=tf.float32,shape=[101,])
#     # w = np.real(eta(x_array,t))
#     w_NN = sess.run(NN_time(x_tf,t_tf,W,b))
#     # erreur += sum((w-w_NN)**2)
#     plt.plot(w_NN,x_array,label="Result from PINN at t = "+str(round(t,3)))
#     # plt.plot(w,x_array,label="Result from theory at t = "+str(round(t,3)))
#     plt.title("Movement of the pipe")
#     plt.xlim([-1,1])
#     plt.ylim([0,1])
#     plt.xlabel("X axis")
#     plt.ylabel("Y axis")
#     plt.legend()
#     plt.show()
# print(erreur/101**2)

z,x = np.loadtxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\XYZT_tip_18Hz1.txt")
start_stop = np.loadtxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\StartStop_2D.txt")
file_id = np.where(start_stop[:,2]==18)[0]
case_id = np.where(start_stop[:,3]==1)[0]
frame_id = np.intersect1d(file_id, case_id)
start = int(start_stop[frame_id,0])
stop = int(start_stop[frame_id,1])
t_plots = np.linspace(0,(stop-start)/150,stop-start)
x_PINN = sess.run(NN_time(tf.constant(z[start:stop],dtype=tf.float32,shape=[stop-start,]),tf.constant(t_plots,dtype=tf.float32,shape=[stop-start,]),W,b))
plt.plot(t_plots,x[start:stop],label="Pipe position along time from the video")
plt.plot(t_plots,x_PINN,label="Pipe position along time from the PINN")
plt.title("Comparison between PINN and the videos")
plt.xlabel("Time in seconds")
plt.ylabel("Deflection in meters")
plt.legend()
plt.show()

np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\TXXNN_Tip_18Hz_2.txt",np.array([t_plots,x[start:stop],x_PINN]))
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\FlowOpt_Opt_18Hz_2.txt",np.array([count,U_list_Adam]))
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\Training_loss_1_18Hz_2.txt",np.array([count,Loss_list_Adam_1]))
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\Training_loss_2_18Hz_2.txt",np.array([count,Loss_list_Adam_2]))
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\Validation_Loss_1_18Hz_2.txt",np.array([count,Validation_loss_list_Adam_1]))
np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\Validation_Loss_2_18Hz_2.txt",np.array([count,Validation_loss_list_Adam_2]))

W_array = sess.run(W)
b_array = sess.run(b)
for i in range(len(layers)-1):
    np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\W_18Hz_2"+str(i)+".txt",W_array[i])
    np.savetxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\B_18Hz_2"+str(i)+".txt",b_array[i])
    
W_array_test = np.empty(3,dtype=object)
b_array_test = np.empty(3,dtype=object)

for i in range(0,len(layers)-1):
    W_array_test[i] = np.loadtxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\W_18Hz_2"+str(i)+".txt")
    b_array_test[i] = np.loadtxt("C:\\Users\Morgan\Videos\Videos_Pipe_0.25in_46cm\B_18Hz_2"+str(i)+".txt")

W_new,b_new = restore_one_NN([2,20,20,1], W_array_test, b_array_test)
sess.run(NN_time(tf.constant(0.4,dtype=tf.float32,shape=(1,)),tf.constant(1.2,dtype=tf.float32,shape=(1,)),W,b))
sess.run(NN_time(tf.constant(0.4,dtype=tf.float32,shape=(1,)),tf.constant(1.2,dtype=tf.float32,shape=(1,)),W_new,b_new))
