#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np


# A two layered neural network is used that takes 14 days worth of crypto-currency data
class NeuralNet():
     
        
    def __init__(self, layers=[14,10,1], learning_rate=0.1, iterations=10000):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.inputData = None
        self.outputGoal = None
                
    def init_weights(self):
        
        np.random.seed(1) 
        self.params['W1'] = np.random.randn(self.layers[0], self.layers[1]) 
        self.params['b1'] = np.random.randn(self.layers[1],)
        self.params['W2'] = np.random.randn(self.layers[1],self.layers[2]) 
        self.params['b2'] = np.random.randn(self.layers[2],)
    
    def relu(self,Z):
       
        return np.maximum(0,Z)

    def dRelu(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x


    def sigmoid(self,Z):
        
        return 1/(1+np.exp(-Z))

    def lossfunc(self,y, yhat):
    
        # a difference squared loss function is used for this net
        lossf = (y-yhat)**2
        
    
        return lossf

    def forward_propagation(self):
        
        
        Z1 = np.dot(self.inputData, self.params['W1'] ) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.params['W2'],) + self.params['b2']
        yhat = self.sigmoid(Z2)
        loss = self.lossfunc(self.outputGoal,yhat)

        # save calculated parameters     
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1

        return yhat,loss

    def back_propagation(self,yhat):
        
        # derivatives from the loss function calculated
        y_inv = 1 - self.outputGoal
        yhat_inv = 1 - yhat

        
        dl_wrt_yhat = -2*(self.outputGoal-yhat)
        dl_wrt_sig = yhat * (yhat_inv)
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = np.dot(dl_wrt_z2, self.params['W2'].T)
        dl_wrt_w2 = dl_wrt_z2*self.params['A1'].T
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)

        dl_wrt_z1 = np.dot(dl_wrt_A1, self.dRelu(self.params['Z1']))
        dl_wrt_w1 = np.dot(dl_wrt_z1,self.inputData.T)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)

        #update the weights and bias

        
        self.params['W1'] = (self.params['W1'].transpose() - (self.learning_rate * dl_wrt_w1)).transpose()
        self.params['W2'] = (self.params['W2'].transpose() - (self.learning_rate * dl_wrt_w2)).transpose()
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2

    def fit(self, X, y):
        # trains the neural net with input data, X, and goal output, y 
        self.inputData = X
        self.outputGoal = y
        self.init_weights() #initialize weights and bias


        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)

    def predict(self, X):
        #neural net's prediction based on input, X
        Z1 = np.dot(X, self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
        p = self.sigmoid(Z2)
        if p > 0.5:
            pred= "Crypto closing price for the day will increase"
        else:
            pred = "Crypto closing price for the day will decrease"
        return pred 


# In[64]:


# instantiate neural net
Net = NeuralNet()


# In[65]:


import csv 
import pandas as pd

# Extracting and organising data to train and test neural net


head = [ 'Unix Timestamp','Date', 'Symbol','Open','High','Low','Close','Volume BTC','Volume USD']

coin_data = pd.read_csv('/Users/milanoreyneke/Downloads/HitBTC_BTCUSD_d.csv', sep=',', names=head)
coin_data


difference = coin_data['Open'] - coin_data['Close']

for i in range(len(difference)):
    if difference[i] > 0:
        difference[i] = 1
    else:
        difference[i] = 0
        

twoWeekData = []
totalData = []
for i in range(26):
    for j in range(14):
        twoWeekData.append((coin_data['Open'][14*i + j]/100)) # dividing all data by 100 to prevent stack overflow
        totalData.append(twoWeekData)
    twoWeekData = []
    
    
twoWeekDiff = []
for i in range(26):
    twoWeekDiff.append(difference[14*i])
    
# crypto-currency prices training data 
trainData = list(zip(totalData, twoWeekDiff))          


# In[66]:


#Training the net with crypto-currency data

for i in range(len(trainData)):
    inputD = np.array(trainData[i][0])
    goal = trainData[i][1]
    Net.fit(inputD, goal)


# In[67]:


#Testing net on data from the training set

print(Net.predict(trainData[0][0]))


# In[ ]:




