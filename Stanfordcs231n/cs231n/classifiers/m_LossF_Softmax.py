
# coding: utf-8

# In[3]:


from __future__ import print_function
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from random import randrange

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[4]:


### Load the raw CIFAR-10 data.
cifar10_dir = '/Users/xiaoxiaoma/cifar-10-batches-py'

# Cleaning up variables to prevent loading data multiple times (which \
# may cause memory issue)
try:
    del X_train, y_train
    del X_test, y_test
    print('Clear previously loaded data.')
except:
    pass
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)


### Subsample the data for more efficient code execution in this exercise
mask = 0
num_train = 500
num_test = 100
num_val = 100
num_dev = 50

mask = range(num_train, num_train + num_val)
X_val = X_train[mask]
y_val = y_train[mask]

mask = range(num_train)
X_train = X_train[mask]
y_train = y_train[mask]

mask = np.random.choice(num_train, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]


### Reshape the image data into rows
# print (X_train.shape[0])
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))


### Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
# print(mean_image[:10]) # print a few of the elements
# plt.figure(figsize=(4,4))
# plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) 
# visualize the mean image
# plt.show()

# second: subtract the mean image from train and test data
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image

# third: append the bias dimension of ones (i.e. bias trick) so that our 
# SVM only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)

# generate a random SVM weight matrix of small numbers, W, shape(num_pixels + 1, num_classes)
def init_W(X, y):
    W = np.random.rand(X.shape[1], np.max(y) + 1) * 0.01
    return W


# In[66]:


def softmax_loss_naive(X, y, W, reg):
    """
    Softmax loss function, naive implementation (with loops)

  Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros_like(W)  #dW.shape (num_pixels+1, num_classes)
######################################################################
# Compute the softmax loss and its gradient using explicit loops.
# Store the loss in loss and the gradient in dW. If you are not careful
# here, it is easy to run into numeric instability. Don't forget the 
# regularization!                                                   
#######################################################################    
##--------------------- written by me start ---------------------##    
    num_classes = W.shape[1]
    num_samples = X.shape[0]

# L_i=−log(e^(f(y[i])+logC)/∑j e^(f(j)+logC)  
    for i in range(num_samples):
        probs = []
        scores = X[i].dot(W) # scores.shape -->(1,C) or should say, (C,)?
        scores -= np.max(scores) # to imporve Numeric stability/avoid potential blowup
        probs = np.exp(scores)/ np.sum(np.exp(scores))
        correct_prob = probs[y[i]]       
        loss += - np.log(correct_prob)
        
        probs[y[i]] -= 1
        dW += np.dot(X[i].reshape(len(X[i]),1), probs.reshape(1, len(probs)))
        
    loss /= num_samples
    loss += 0.5 * reg * np.sum(W * W)
    
    dW /= num_samples
    dW += reg * W

##--------------------- written by me end -----------------------##    
    return loss, dW


# In[63]:


def softmax_loss_vectorized(X, y, W, reg):
    """
    Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
    """  
    
    loss = 0.0
    dW = np.zeros_like(W)
    
############################################################################
# Compute the softmax loss and its gradient using no explicit loops.
# Store the loss in loss and the gradient in dW. If you are not careful
# here, it is easy to run into numeric instability. Don't forget the 
# regularization!                             
####################################################################### 
##--------------------- written by me start ---------------------##    

    num_samples = X.shape[0]
    probs = [] # probs.shape (num_samples, num_classes)
    
    # Loss = -log(probability of y_i) , in which:
    # probability of k = exp(f_k)/ (∑j exp(f_j))
    scores = np.dot(X, W)    # scores.shape (num_samples, num_classes)
    scores -= np.max(scores, axis = 1, keepdims = True)
    probs = np.exp(scores) / np.sum(np.exp(scores), axis = 1, keepdims = True) 
    # probs.shape (num_samples, num_classes)
    correct_logprobs = -np.log(probs[np.arange(X.shape[0]), y]) 
    loss = np.sum(correct_logprobs) / num_samples
    loss += 0.5 * reg * np.sum(W*W)
    
    # gradient = probability_j - (1 for j == y_i, 0 for others) 
    probs [np.arange(X.shape[0]),y] -= 1
    
    dW = np.dot(X.T, probs)/num_samples
    dW += reg * W

##--------------------- written by me end -----------------------##      
    return loss, dW


# In[ ]:


W_dev = init_W(X_dev, y_dev)


# In[67]:


loss_n, grad_n = softmax_loss_naive(X_dev, y_dev, W_dev, 0.005)
print('loss_n: %f' % (loss_n, ))
print('grad: {0}'.format(grad_n[:2,:5]))
loss_v, grad_v = softmax_loss_vectorized(X_dev, y_dev, W_dev, 0.005)
print('loss_v: %f' % (loss_v, ))
print('grad: {0}'.format(grad_v[:2,:5]))


# In[ ]:


from cs231n.gradient_check import grad_check_sparse


# In[68]:


f = lambda w: softmax_loss_naive(X_dev, y_dev, w, 0.0)[0]
grad_numerical = grad_check_sparse(f, W_dev, grad_n)


# In[65]:


f = lambda w: softmax_loss_vectorized(X_dev, y_dev, w, 0.0)[0]
grad_numerical = grad_check_sparse(f, W_dev, grad_v)


# In[70]:


import time
tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(X_dev, y_dev, W_dev,0.005)
toc = time.time()
print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))

tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(X_dev, y_dev,                                                        W_dev, 0.005)
toc = time.time()
print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# The losses should match but your vectorized implementation should be \
# much faster.
print('difference: %f' % (loss_naive - loss_vectorized))

