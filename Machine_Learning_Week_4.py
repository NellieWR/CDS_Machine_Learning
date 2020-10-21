#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io
import scipy.optimize
import matplotlib.pyplot as plt
mat = scipy.io.loadmat('mnistAll.mat')
import numpy as np
import time


# In[2]:


mnist = mat['mnist']
test1 = mnist[0, 0]
training_images=mat['mnist'][0][0][0]
test_images=mat['mnist'][0][0][1]
training_labels=mat['mnist'][0][0][2]
test_labels=mat['mnist'][0][0][3]


# In[3]:


training_images_3 = training_images[:, :, np.where(training_labels==3)[0]] #Take all 3's
training_images_7 = training_images[:, :, np.where(training_labels==7)[0]] #Take all 7's
n3 = training_images_3.shape[2] #Number of 3's
n7 = training_images_7.shape[2] #Number of 7's
training_images_3 = np.reshape(training_images_3, (784, n3)) #Stretch images.
training_images_7 = np.reshape(training_images_7, (784, n7)) #Stretch images.
training_images_37 = np.append(training_images_3, training_images_7, axis=1) #Stick the 3's and 7's together.
training_labels_37 = np.append(np.zeros((1, n3)), np.ones((1, n7))) #Binary labels.
training_N = training_images_37.shape[1] #Number of images.
training_images_37 = training_images_37/255 #Renormalise.
training_images_37 = np.append(training_images_37, np.ones((1, training_N)), axis=0)
n = training_images_37.shape[0] #Pixels per image.


# In[4]:


test_images_3 = test_images[:, :, np.where(test_labels==3)[0]] #Take all 3's
test_images_7 = test_images[:, :, np.where(test_labels==7)[0]] #Take all 7's
n3 = test_images_3.shape[2] #Number of 3's
n7 = test_images_7.shape[2] #Number of 7's
test_images_3 = np.reshape(test_images_3, (784, n3)) #Stretch images.
test_images_7 = np.reshape(test_images_7, (784, n7)) #Stretch images.
test_images_37 = np.append(test_images_3, test_images_7, axis=1) #Stick the 3's and 7's together.
test_labels_37 = np.append(np.zeros((1, n3)), np.ones((1, n7))) #Binary labels.
test_N = test_images_37.shape[1] #Number of images.
test_images_37 = test_images_37/255 #Renormalise.
test_images_37 = np.append(test_images_37, np.ones((1, test_N)), axis=0)


# In[5]:


def prob1(weights, images):
    prodsum = np.sum(images*weights, axis=0)
    p1 = (1+np.exp(-1*prodsum))**-1
    return p1


# In[6]:


def loglikelihood(labels, p1, weights, images, N):
    E = 0
    for ii in range(N):
        if (labels[ii]==0):
            E = E-1/N*(1-labels[ii])*np.log(1-p1[ii])
        if (labels[ii]==1):
            E = E-1/N*labels[ii]*np.log(p1[ii])
    return E


# In[7]:


def loglikelihood_weight_decay(labels, p1, weights, N, n):
    E = 0
    for ii in range(labels.shape[0]):
        if (labels[ii]==0):
            E = E-1/N*(1-labels[ii])*np.log(1-p1[ii])
        if (labels[ii]==1):
            E = E-1/N*labels[ii]*np.log(p1[ii])
    E_weight_decay = 0.1/(2*n)*np.sum(weights**2)
    return E


# In[8]:


def gradient(p1, images, labels, N):
    grad = 1/N*(np.sum((p1-labels)*images, axis=1))
    return grad


# In[9]:


def gradient_weight_decay(p1, weights, images, labels, N, n):
    grad_normal = 1/N*(np.sum((p1-labels)*images, axis=1))
    grad_decay = 0.1/n*weights
    grad = np.reshape(grad_normal, (785, 1))+grad_decay
    return grad


# In[10]:


def hessian(p1, images, N, n):
    H = np.zeros((n, n))
    yprod = p1*(1-p1)
    for ii in range(n):
        for jj in range(ii, n): #This range is used because the Hessian is symmetric
            H[ii, jj] = np.sum(images[ii, :]*yprod*images[jj, :])
            H[jj, ii] = H[ii, jj]
    H = H/N
    return H


# In[11]:


def hessian_weight_decay(p1, images, N, n):
    H = np.zeros((n, n))
    yprod = p1*(1-p1)
    for ii in range(n):
        for jj in range(ii, n): #This range is used because the Hessian is symmetric
            H[ii, jj] = np.sum(images[ii, :]*yprod*images[jj, :])
            H[jj, ii] = H[ii, jj]
    H = H/N + np.identity(n)*0.1/n
    return H


# In[12]:


def newton(images, labels, weights, N, n):
    p1 = prob1(weights, images)
    E = loglikelihood_weight_decay(labels, p1, weights, N, n)
    grad = gradient_weight_decay(p1, weights, images, labels, N, n)
    H = hessian_weight_decay(p1, images, N, n)
    
    H_inv = np.linalg.inv(H)
    weight_change = -1*np.matmul(H_inv, grad)
    return weight_change, E


# In[13]:


def to_optimise(gamma, weights, labels, images, N):
    p1 = prob1(weights, images)
    d = -1*gradient(p1, images, labels, N)
    return loglikelihood(labels, p1, weights+gamma*d, images, N)
#labels, weights, images, N


# In[14]:


def linesearch(images, labels, weights, N, n):
    p1 = prob1(weights, images)
    d = -1*gradient(p1, images, labels, N)
    
    def to_optimise(gamma, weights, labels, images, N):
        p1 = prob1(weights, images)
        d = -1*gradient(p1, images, labels, N)
        return loglikelihood(labels, p1, weights+gamma*d, images, N)
    
    #result = scipy.optimize.minimize(to_optimise, args=(weights, training_labels_37, training_images_37, training_N), x0=1)
    result = scipy.optimize.minimize(to_optimise, x0=1, args=(weights, labels, images, N))
    gamma = result.x
    weight_change = gamma*d
    weight_change = np.reshape(weight_change, (785, 1))
    return weight_change, d


# In[15]:


def polakribiere(labels, weights_0, weights_1, images, N):
    p1_0 = prob1(weights_0, images)
    p1_1 = prob1(weights_1, images)
    grad_0 = gradient(p1_0, images, labels, N)
    grad_1 = gradient(p1_1, images, labels, N)
    grad_diff = grad_1-grad_0
    beta_num = np.inner(grad_diff, grad_1)
    beta_denom = np.linalg.norm(grad_0)**2
    beta = beta_num/beta_denom
    return beta


# In[16]:


def conjugate_grad(labels, weights_0, weights_1, d_0, images, N):
    p1_0 = prob1(weights_0, images)
    p1_1 = prob1(weights_1, images)
    #print("In conjugate gradient, p1_1 has shape: ", p1_1.shape)
    #print("In conjugate gradient, labels has shape: ", labels.shape)
    beta = polakribiere(labels, weights_0, weights_1, images, N)
    grad_1 = gradient(p1_1, images, labels, N)
    d_1 = -1*grad_1+beta*d_0
    
    def to_optimise(gamma, weights_1, labels, images, N, d_1, p1_1):
        #print("In to_optimise, p1_1 has shape: ", p1_1.shape)
        #print("In to_optimise, labels has shape: ", labels.shape)
        return loglikelihood(labels, p1_1, weights_1+gamma*d_1, images, N)
    
    result = scipy.optimize.minimize(to_optimise, x0=1, args=(weights_1, labels, images, N, d_1, p1_1))
    gamma = result.x
    weight_change = gamma*d_1
    weight_change = np.reshape(weight_change, (785, 1))
    return weight_change, weights_1, d_1
#labels, p1, weights, images, N


# In[35]:


def stochasticgradient(weights, labels, images, N):
    sample_N = round(N/100)
    sample_ind = np.random.choice(N, size = (sample_N, 1), replace=False)
    images_sample = np.take(images, sample_ind, axis = 1)
    images_sample = np.reshape(images_sample, (785,sample_N))
    labels_sample = np.take(labels, sample_ind, axis = 0)
    labels_sample = np.reshape(labels_sample, (sample_N, ))
    p1 = prob1(weights, images_sample)
    grad = gradient(p1, images_sample, labels_sample, N)
    weights_change = -1*grad
    weights_change = np.reshape(weights_change, (785, 1))
    return weights_change


# In[19]:


# Newton's method evaluated.
weights = 0.1*np.random.normal(0, 1, (785, 1))
#weights = np.random.choice(a=(-1, 1), size = (785, 1))
E_training = np.zeros((11, 1))
E_test = np.zeros((11,1))

start_time = time.time()

for ii in range(10):
    p1_test = prob1(weights, test_images_37)
    E_test[ii] = loglikelihood_weight_decay(test_labels_37, p1_test, weights, test_N, n)
    weights_change, E_training[ii] = newton(training_images_37, training_labels_37, weights, training_N, n)
    weights = weights + weights_change

p1_training_10 = prob1(weights, training_images_37)
E_training[10] = loglikelihood_weight_decay(training_labels_37, p1_training_10, weights, training_N, n)
p1_test_10 = prob1(weights, test_images_37)
E_test[10] = loglikelihood_weight_decay(test_labels_37, p1_test_10, weights, test_N, n)

end_time = time.time()
elapsed_time = end_time-start_time


# In[20]:


#Newton's method plots.
iterations = np.linspace(0, 10, 11)
#plt.plot(iterations, E_training, iterations, E_test)
plt.plot(iterations, E_training, label = 'Training set')
plt.plot(iterations, E_test, label = "Test set")
plt.xlabel("Iterations")
plt.ylabel("Log-likelihood")
plt.legend()
plt.savefig("Newton_method.png")
plt.show()
print("Final training error: ", E_training[10])
print("Final test error: ", E_test[10])
print("Elapsed time: ", elapsed_time)


# In[23]:


#Newton's method plots.
np.amax(E_test)
print(E_test)
plt.plot(iterations[4:10], E_test[4:10])
plt.xlabel("Iterations")
plt.ylabel("Log-likelihood test set")
plt.savefig("Test_error_zoom.png")
plt.show()


# In[27]:


#Line search method evaluated.
weights = np.random.normal(0, 1, (785, 1))
E_training_ls = np.zeros((251, 1))
E_test_ls = np.zeros((251, 1))
start_time = time.time()

p1_training = prob1(weights, training_images_37)
E_training_ls[0] = loglikelihood(training_labels_37, p1_training, weights, training_images_37, training_N)
p1_test = prob1(weights, test_images_37)
E_test_ls[0] = loglikelihood(test_labels_37, p1_test, weights, test_images_37, test_N)

for ii in range(250):
    #print(ii)
    #print(weights.shape)
    
    weights_change, _ = linesearch(training_images_37, training_labels_37, weights, training_N, n)
    weights = weights+weights_change
    #print(weights.shape)
    p1_training = prob1(weights, training_images_37)
    E_training_ls[ii+1] = loglikelihood(training_labels_37, p1_training, weights, training_images_37, training_N)
    p1_test = prob1(weights, test_images_37)
    E_test_ls[ii+1] = loglikelihood(test_labels_37, p1_test, weights, test_images_37, test_N)

end_time = time.time()
elapsed_time = end_time-start_time


# In[29]:


#Line search method plots.
iterations = np.linspace(0, 250, 251)
#plt.plot(iterations, E_training, iterations, E_test)
plt.plot(iterations, E_training_ls, label = 'Training set')
plt.plot(iterations, E_test_ls, label = "Test set")
plt.xlabel("Iterations")
plt.ylabel("Log-likelihood")
plt.legend()
plt.savefig("Line_search.png")
plt.show()
print("Final training error: ", E_training_ls[250])
print("Final test error: ", E_test_ls[250])
print("Elapsed time: ", elapsed_time)


# In[40]:


# Conjugate gradient method evaluation.

weights = np.random.normal(0, 1, (785, 1))
E_training_cg = np.zeros((151, 1))
E_test_cg = np.zeros((151, 1))
start_time = time.time()

p1_training = prob1(weights, training_images_37)
E_training_cg[0] = loglikelihood(training_labels_37, p1_training, weights, training_images_37, training_N)
p1_test = prob1(weights, test_images_37)
E_test_cg[0] = loglikelihood(test_labels_37, p1_test, weights, test_images_37, test_N)

#First iteration of the conjugate gradient method is simply the line search algorithm.
weights_change, d_0 = linesearch(training_images_37, training_labels_37, weights, training_N, n)
weights_0 = weights
weights_1 = weights+weights_change
p1_training = prob1(weights_1, training_images_37)
E_training_cg[1] = loglikelihood(training_labels_37, p1_training, weights, training_images_37, training_N)
p1_test = prob1(weights_1, test_images_37)
E_test_cg[1] = loglikelihood(test_labels_37, p1_test, weights, test_images_37, test_N)

#labels, weights_0, weights_1, d_0, images, N
for ii in range(1, 150):
    weights_change, weights_0, d_0 = conjugate_grad(training_labels_37, weights_0, weights_1, d_0, training_images_37, training_N)
    weights_1 = weights_0+weights_change
    p1_training = prob1(weights_1, training_images_37)
    E_training_cg[ii+1] = loglikelihood(training_labels_37, p1_training, weights, training_images_37, training_N)
    p1_test = prob1(weights_1, test_images_37)
    E_test_cg[ii+1] = loglikelihood(test_labels_37, p1_test, weights, test_images_37, test_N)

end_time = time.time()
elapsed_time = end_time-start_time


# In[41]:


#Conjugate gradient method plots.
iterations = np.linspace(0, 150, 151)
#plt.plot(iterations, E_training, iterations, E_test)
plt.plot(iterations, E_training_cg, label = 'Training set')
plt.plot(iterations, E_test_cg, label = "Test set")
plt.xlabel("Iterations")
plt.ylabel("Log-likelihood")
plt.legend()
plt.savefig("Conjugate_gradient.png")
plt.show()
print("Final training error: ", E_training_cg[150])
print("Final test error: ", E_test_cg[150])
print("Elapsed time: ", elapsed_time)


# In[26]:


weights = np.random.normal(0, 1, (785, 1))
stochastic = stochasticgradient(weights, training_labels_37, training_images_37, training_N)


# In[43]:


#Stochastic gradient method evaluated.
weights = np.random.normal(0, 1, (785, 1))
print(weights.shape)
E_training_sg = np.zeros((5001, 1))
E_test_sg = np.zeros((5001, 1))
start_time = time.time()

p1_training = prob1(weights, training_images_37)
E_training_sg[0] = loglikelihood(training_labels_37, p1_training, weights, training_images_37, training_N)
p1_test = prob1(weights, test_images_37)
E_test_sg[0] = loglikelihood(test_labels_37, p1_test, weights, test_images_37, test_N)

for ii in range(5000):
    print(ii)
    weights_change = stochasticgradient(weights, training_labels_37, training_images_37, training_N)
    weights = weights+weights_change
    p1_training = prob1(weights, training_images_37)
    E_training_sg[ii+1] = loglikelihood(training_labels_37, p1_training, weights, training_images_37, training_N)
    p1_test = prob1(weights, test_images_37)
    E_test_sg[ii+1] = loglikelihood(test_labels_37, p1_test, weights, test_images_37, test_N)

end_time = time.time()
elapsed_time = end_time-start_time


# In[44]:


#Stochastic gradient method plots.
iterations = np.linspace(0, 5000, 5001)
#plt.plot(iterations, E_training, iterations, E_test)
plt.plot(iterations, E_training_sg, label = 'Training set')
plt.plot(iterations, E_test_sg, label = "Test set")
plt.xlabel("Iterations")
plt.ylabel("Log-likelihood")
plt.legend()
plt.savefig("Stochastic_gradient.png")
plt.show()
print("Final training error: ", E_training_sg[5000])
print("Final test error: ", E_test_sg[5000])
print("Elapsed time: ", elapsed_time)

