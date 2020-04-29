##############################################################################
#
#    Mahnoor Anjum
#    manomaq@gmail.com
#    
#    References:
#        SuperDataScience,
#        Official Documentation
#
#
##############################################################################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = [[4,1.5,1],
          [3,1,0],
          [5,1.5,1],
          [4,1,0], 
          [4.5,0.5,1],
          [3,0.5,0],
          [6.5,1,1],
          [1,1,0]]
test = [4.5, 1] 

for i in range(len(data)):
    point = data[i]
    if point[2]==1:
        plt.scatter(point[0],point[1], c = 'r')
    else:
        plt.scatter(point[0],point[1], c = 'b') 
plt.grid()
plt.clf()


def sigmoid(x):
    return 1/(1+ np.exp(-x))
def sigmoid_h(x):
    return sigmoid(x) * (1-sigmoid(x))


x = np.linspace(-5,5,100)
y = sigmoid(x)
y1 = sigmoid_h(x)
plt.plot(x,y, 'r', label="Sigmoid")
plt.plot(x,y1, 'k', label="Derivative of Sigmoid")
plt.legend()
plt.show()
plt.clf()

# TRAINING
costs = []
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()
for i in range(100000):
    lr = 1.2
    index = np.random.randint(len(data))
    point = data[index]
    net = point[0] * w1 + point[1]* w2 + b
    # NET = ( X1 x W1 ) + ( X2 x W2 ) + B
    pred = sigmoid(net)
    # PRED = SIGMOID ( ( X1 x W1 ) + ( X2 x W2 ) + B )
    target = point[2]
    cost = np.square(pred - target)
    # COST = ( PRED - TARGET )^2
    # COST = ( SIGMOID ( ( X1 x W1 ) + ( X2 x W2 ) + B ) - TARGET )^2
    dcost_dpred = 2*(pred-target)
    # COST_wrt_PRED = 2 x ( PRED - TARGET) 
    dpred_dnet = sigmoid_h(net)
    # PRED_wrt_NET = derivative of sigmoid ( NET )
    dnet_dw1 = point[0]
    # NET_wrt_W1 = X1
    dnet_dw2 = point[1]
    # NET_wrt_W2 = X2
    dnet_db = 1
    # NET_wrt_B = 1
    
    #CHAIN RULE
    dcost_dw1 = dcost_dpred * dpred_dnet * dnet_dw1
    dcost_dw2 = dcost_dpred * dpred_dnet * dnet_dw2
    dcost_db = dcost_dpred * dpred_dnet * dnet_db
    
    #UPDATE
    w1 = w1 - lr* dcost_dw1
    w2 = w2 - lr* dcost_dw2
    b = b - lr* dcost_db
    
    costs.append(cost)
plt.plot(costs)