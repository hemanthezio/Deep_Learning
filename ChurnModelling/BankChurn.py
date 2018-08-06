# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 11:10:10 2018

@author: Hemanth kumar
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from DeepNetwork import DeepNetwork

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, [13]].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train=X_train.T
X_test=X_test.T
y_train=y_train.T
y_test=y_test.T

# Training the model
n_x=len(X_train)
af=['tanh','sigmoid']
layer_dim=[n_x,6,8,1,]
layer_size=len(layer_dim)

#Initializing Deep Network
MyNetwork=DeepNetwork(X_train,y_train)


start = time.clock()
para,J_log=MyNetwork.network(X_train,y_train,af,layer_dim,epoch=500,batch_size=25,alpha=0.05,plot=False)
end= time.clock()
elapsed = (end - start)
print("Elapsed time:\n",elapsed)

# Plotting cost v/s number of iterations
plt.title('Cost V/S Epochs')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.plot(range(len(J_log)), J_log)
plt.show()
print(layer_dim)


# Testing the model
from sklearn.metrics import accuracy_score
y_pred=MyNetwork.predict(X_train,layer_size,para,af)
accuracy = accuracy_score(y_train.T,y_pred.T)
print('Train Accuracy Score:',accuracy*100,'%')

y_pred=MyNetwork.predict(X_test,layer_size,para,af)
accuracy = accuracy_score(y_test.T,y_pred.T)
print('Test Accuracy Score:',accuracy*100,'%')

#K-Fold cross validation b=25 ep=500 rmsprop
scores=MyNetwork.k_fold_crossval(af,layer_dim,epoch=500,batch_size=25,alpha=0.05,k=10)
print("Accuracy of the model from K-FOLD CROSS VALIDATION:",np.mean(scores)*100)
