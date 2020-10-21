#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal


# In[2]:


pd_data = pd.read_table("faithfuldata.txt")
faithful = pd_data.to_numpy() #data(:, 0) = eruptions, data(:, 1) = waiting, (272, 2)

# Modifying the data
faithful[:, 0] = (faithful[:, 0]-np.mean(faithful[:, 0]))/np.std(faithful[:, 0])
faithful[:, 1] = (faithful[:, 1]-np.mean(faithful[:, 1]))/np.std(faithful[:, 1])


# In[3]:


def em(K, N, mu1, mu2, sigma1, sigma2, pi, data):
    
    # Some processing
    mu1 = np.reshape(mu1, (K, ))
    mu2 = np.reshape(mu2, (K, ))
    # E step
    normal1 = multivariate_normal.pdf(data, mu1, sigma1)
    #print("normal1: ", normal1)
    normal2 = multivariate_normal.pdf(data, mu2, sigma2)
    #print("normal2: ", normal2)
    normal = np.append(np.reshape(normal1, (N, 1)), np.reshape(normal2, (N, 1)), axis=1)
    #print("normal: ", normal)
    pinormalsum = np.reshape(np.sum(pi*normal, axis=1), (N, 1))
    #print("pinormalsum: ", pinormalsum)
    q = (pi*normal)/pinormalsum #(272, 2)
    #print("q: ", q)
    
    # M step
    
    newsigma1 = np.zeros((2, 2))
    newsigma2 = np.zeros((2, 2))
    r = np.reshape(np.sum(q, axis=0), (1, K)) #(1, K)
    #print("r: ", r)
    newpi = r/N #(1, K)
    #print("pi: ", pi)
    newmu1 = np.reshape(np.sum(np.reshape(q[:, 0], (N, 1))*data, axis=0), (1, K))/r #(1, K)
    #print("mu1: ", mu1)
    newmu2 = np.reshape(np.sum(np.reshape(q[:, 1], (N, 1))*data, axis=0), (1, K))/r #(1, K)
    #print("mu2: ", mu2)
    for ii in range(K): #sigma1.shape = (K, K)
        for jj in range(K):
            newsigma1[ii, jj] = np.sum(q[:, 0]*data[:, ii]*data[:, jj])/r[0, 0]-newmu1[0, ii]*newmu1[0, jj]
    for ii in range(K): #sigma1.shape = (K, K)
        for jj in range(K):
            newsigma2[ii, jj] = np.sum(q[:, 1]*data[:, ii]*data[:, jj])/r[0, 1]-newmu2[0, ii]*newmu2[0, jj]
    
    return newmu1, newmu2, newsigma1, newsigma2, newpi


# In[6]:


# Initialising of variables.
K = 2
N = faithful.shape[0]
mu1 = np.array([-1.0, 1.0]) 
mu2 = np.array([1.0, -1.0])
sigma1 = np.identity(K)
sigma2 = np.identity(K)
pi = np.random.random((1, K))
pi = pi/np.sum(pi)
print(pi[0, 0]+pi[0, 1])
q = np.zeros((K, 1))

plt.scatter(faithful[:, 0], faithful[:, 1])
plt.scatter(mu1[0], mu1[1])
plt.scatter(mu2[0], mu2[1])

eruptions = np.linspace(np.amin(faithful[:, 0]), np.amax(faithful[:, 0]), 1000)
waiting = np.linspace(np.amin(faithful[:, 1]), np.amax(faithful[:, 1]), 1000)
X, Y = np.meshgrid(eruptions, waiting)
  
pos = np.array([X.flatten(), Y.flatten()]).T
  
normalplot1 = multivariate_normal(mean = np.reshape(mu1, (2, )), cov = sigma1)
normalplot2 = multivariate_normal(mean = np.reshape(mu2, (2, )), cov = sigma2)
    
    
plt.contour(eruptions, waiting, normalplot1.pdf(pos).reshape(1000, 1000), levels = 1)
plt.contour(eruptions, waiting, normalplot2.pdf(pos).reshape(1000, 1000), levels = 1)
plt.xlabel("Eruptions")
plt.ylabel("Waiting")
plt.title("Iteration: 0")
plt.savefig("EM_iteration_0.png")
plt.clf()

#Iterations of EM
for ii in range(30):
    print(ii)
    mu1, mu2, sigma1, sigma2, pi = em(K, N, mu1, mu2, sigma1, sigma2, pi, faithful)
    print("pi: ", pi)
    print("mu1: ", mu1)
    print("mu2: ", mu2)
    print("sigma1: ", sigma1)
    print("sigma2: ", sigma2)
    plt.scatter(faithful[:, 0], faithful[:, 1])
    plt.scatter(mu1[0, 0], mu1[0, 1])
    plt.scatter(mu2[0, 0], mu2[0, 1])
    
    eruptions = np.linspace(np.amin(faithful[:, 0]), np.amax(faithful[:, 0]), 1000)
    waiting = np.linspace(np.amin(faithful[:, 1]), np.amax(faithful[:, 1]), 1000)
    X, Y = np.meshgrid(eruptions, waiting)
    
    pos = np.array([X.flatten(), Y.flatten()]).T
    
    normalplot1 = multivariate_normal(mean = np.reshape(mu1, (2, )), cov = sigma1)
    normalplot2 = multivariate_normal(mean = np.reshape(mu2, (2, )), cov = sigma2)
    
    plt.contour(eruptions, waiting, normalplot1.pdf(pos).reshape(1000, 1000), levels = 1)
    plt.contour(eruptions, waiting, normalplot2.pdf(pos).reshape(1000, 1000), levels = 1)
    plt.xlabel("Eruptions")
    plt.ylabel("Waiting")
    plt.title("Iteration: {n}".format(n=ii+1))
    plt.savefig("EM_iteration_{n}.png".format(n=ii+1))
    plt.clf()


# In[40]:


testdata = np.array([[0.268, 0.717], [-1.771, -1.170], [-0.019, 0.340]])
print("Testdata: ", testdata)
mu1 = np.array([-1.0, 1.0])
print("mu1: ", mu1)
mu2 = np.array([1.0, -1.0])
print("mu2: ", mu2)
sigma1 = np.array([[1, 0], [0, 1]])
sigma2 = sigma1
print("sigma1: ", sigma1)
print("sigma2: ", sigma2)
pi = np.array([[0.451, 0.549]])
print("pi: ", pi)
K = 2
N = 3
mu1, mu2, sigma1, sigma2, pi = em(K, N, mu1, mu2, sigma1, sigma2, pi, testdata)
print("mu1: ", mu1)
print("mu2: ", mu2)
print("sigma1: ", sigma1)
print("sigma2: ", sigma2)
print("pi: ", pi)

