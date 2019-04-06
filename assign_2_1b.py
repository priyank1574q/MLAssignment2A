import numpy as np
import pandas as pd
import time as t
import os
import sys

train_read = sys.argv[1]
train_path = os.path.abspath(train_read)
path_train = os.path.dirname(train_path)
os.chdir(path_train)

test_read = sys.argv[2]
test_path = os.path.abspath(test_read)
path_test = os.path.dirname(test_path)
os.chdir(path_test)

x_train = (pd.read_csv(train_read, header = None, na_filter = False, low_memory = False)).values
x_test = (pd.read_csv(test_read, header = None, na_filter = False, low_memory = False)).values
y_train = list(x_train[:,0])
y_test = list(x_test[:,0])
x_train = x_train[:,1:]/255.0
x_test = x_test[:,1:]/255.0

method = 'sigmoid'
batch = 128

def f(l, method):
    if(method == 'relu'):
        l[l<0] = 0
        return(l)
    
    elif(method == 'sigmoid'):
        return(1.0/(1+np.exp(-l)))
    
    elif(method == 'tanh'):
        return(np.tanh(l))

def df(l, method):
    if(method == 'relu'):
        l[l<0] = 0
        l[l>0] = 1
        return(l)
    
    elif(method == 'sigmoid'):
        return(f(l, 'sigmoid')*(1-f(l, 'sigmoid')))
    
    elif(method == 'tanh'):
        return(1 - (np.tanh(l))**2)

def layers(inputs = 1024, outputs = 46, hidden = [100, 100]):
    w = [None]*(len(hidden))
    z = [None]*(len(hidden))    
    bias = [None]*(len(hidden)-1)
    
    w[0] = np.random.uniform(low = 0, high = 0.1, size = (inputs, hidden[0]))
    z[0] = np.random.rand(inputs,1)
    
    for i in range(1,len(hidden)):
        w[i] = np.random.uniform(low = 0, high = 0.1, size = (hidden[i-1],hidden[i]))
        z[i] = np.zeros((hidden[i-1],1))
        bias[i-1] = np.random.rand(hidden[i-1],1)
    
    w.append(np.random.uniform(low = 0, high = 0.1, size = (hidden[len(hidden)-1], outputs)))
    
    z.append(np.zeros((hidden[len(hidden)-1],1)))
    z.append(np.zeros((outputs,1)))
    
    bias.append(np.random.rand(hidden[len(hidden)-1],1))
    bias.append(np.random.rand(outputs,1))
    
    network = [w, z, bias]
    
    return network

def forward(network, ex):
    w = network[0]
    z = network[1]
    bias = network[2]
    
    ex = np.array(ex)
    ex.shape = (len(ex),1)
    
    z[0] = ex
    z[1] = (w[0].T).dot(z[0]) + bias[0]
    
    for i in range(1,len(w)-1):
        z[i+1] = (w[i].T).dot(f(z[i], method)) + bias[i]
    
    z[len(z)-1] = (w[len(w)-1].T).dot(f(z[len(z)-2], 'sigmoid')) + bias[len(bias)-1]
    
    return(network)

# data = layers(inputs = 784, outputs = 10, hidden = [100])

# net1 = forward(data, x_train[0])

# out = [0]*10
# out[y_train[0]] = 1
# out = np.array(out)
# out.shape = (len(out),1)

# w = net1[0]
# z = net1[1]
# bias = net1[2]

# delta = [None]*(len(z)-1)
# del_w = [None]*(len(w))
# del_b = [None]*(len(bias))

# delta[len(delta)-1] = (f(z[len(z)-1], 'sigmoid') - out)*df(z[len(z)-1], 'sigmoid')
# del_w[len(w)-1] = delta[len(delta)-1]*(f(z[len(z)-2], method).T)
# del_b[len(bias)-1] = np.copy(delta[len(delta)-1])
# del_w[0] = delta[0]*f(z[0], method).T
# del_b[0] = (w[1].dot(delta[1]))*df(z[1], method)

def backward(network, out):
    w = network[0]
    z = network[1]
    bias = network[2]
    
    delta = [None]*(len(z)-1)
    del_w = [None]*(len(w))
    del_b = [None]*(len(bias))
    
    delta[len(delta)-1] = (f(z[len(z)-1], 'sigmoid') - out)*df(z[len(z)-1], 'sigmoid')
    del_w[len(w)-1] = delta[len(delta)-1]*(f(z[len(z)-2], method).T)
    del_b[len(bias)-1] = np.copy(delta[len(delta)-1])
    
    for i in range(len(delta)-2,-1,-1):
        delta[i] = (w[i+1].dot(delta[i+1]))*df(z[i+1], method)
        del_w[i] = delta[i]*f(z[i], method).T
        del_b[i] = (w[i+1].dot(delta[i+1]))*df(z[i+1], method)
        
    del_w = np.array(del_w)
    del_b = np.array(del_b)
    chg = [del_w, del_b]
    
    return(chg)

def train(network, x_train, y_train, batch, rate):
    t_w = np.copy(network[0])*0
    t_b = np.copy(network[2])*0
    
    w = network[0]
    z = network[1]
    bias = network[2]
    
    for num_iter in range(1):
        for i in range(0,x_train.shape[0], batch):
            for j in range(i, i+batch):
                net = forward([w, z, bias], x_train[j,:])
                
                out = [0]*46
                out[y_train[j]] = 1
                out = np.array(out)
                out.shape = (len(out),1)
                
                delta_w, delta_b = backward(net, out)
                for k in range(len(t_w)):
                    t_w[k] += delta_w[k].T
                    t_b[k] += delta_b[k]
                    
            for k in range(len(t_w)):
                w[k] -= (t_w[k])*(rate/((batch)*(num_iter+1)**0.5))
                bias[k] -= (t_b[k])*(rate/((batch)*(num_iter+1)**0.5))
    
    return(network)        

# one = layers(inputs = 1024, outputs = 46, hidden = [100, 100])
# two = train(one, x_train, y_train, batch, 0.1)

# ans = []
# for i in range(x_test.shape[0]):
#     c = forward(b, x_test[i])
#     ans.append(np.argmax(f(c[1][2]/max(c[1][2]), 'sigmoid')))

ans = list(np.array(y_train)*0)

out_write = sys.argv[3]
path_out = os.path.abspath(out_write)
out_path = os.path.dirname(path_out)
os.chdir(out_path)
np.savetxt(out_write, ans)

