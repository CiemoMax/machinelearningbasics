
# coding: utf-8

# In[1]:


from __future__ import print_function
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from random import randrange
import time 

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# for ipython, use: %matplotlib osx

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[48]:


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
    W = np.random.rand(X.shape[1], np.max(y) + 1) * 0.0001
    return W


# In[46]:


# Loss functions

def svm_loss_naive(X, y, W, reg = 0.0001):
    """
    Structured SVM loss function, naive implementation (with loops).
    unvectorized. Compute the multiclass svm loss for a single example(x,y)
    
    - x is a column vector representing an image (e.g. 3073 x 1 in 
    CIFAR-10)with an appended bias dimension in the 3073-rd position 
    (i.e. bias trick)
    - y is an integer giving index of correct class (e.g. between 0 and 9)
    - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  
    Inputs:

    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels, (0,C); 
      y[i] = c means that X[i] has label c, where 0 <= c < C.
    - W: A numpy array of shape (D, C) containing weights.
    - reg: (float) regularization strength

    returns:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W 
    """
    
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    delta = 1.0
    num_classes = W.shape[1]
    num_samples = X.shape[0]
    loss = 0.0
    
    for i in range(num_samples):
        scores = X[i].dot(W) # scores.shape -->(1,C) or should say, (C,)?
        correct_class_score = scores[y[i]]
        num_high_margin = 0
        
        for j in range(num_classes):
            if j == y[i]:
                continue
                
            margin = scores[j] - correct_class_score + delta
#             print('margin_sample{0},class{1}: {2}'.format(i,j,margin)) #debug
            if margin > 0:
                loss += margin
                dW.T[j] += X[i]
                num_high_margin += 1
        
#         print('loss after loop{0}: {1}'.format(i,loss)) #debug
        
        dW.T[y[i]] += - num_high_margin * X[i]
#         print('dW after loop{1}: {0}'.format(dW,i)) #debug
  
  # Compute the gradient (analytic gradient) of the loss function and 
  # store it dW.     
  # it may be simpler to compute the derivative at the same time that
  # the loss is being computed.    

# Right now the loss is a sum over all training examples, but we want it
# to be an average instead so we divide by num_train. --> 'data loss'
# L=1/N * ∑i ∑j≠y[i] [max(0, f(x_i;W)_j - f(x_i, W)_y[j] + Δ]
# f(x_i;W)= Wx_i
    loss /= num_samples
    dW /= num_samples

# Add regularization, R(W)), to the loss. R(W)=lambda * ∑k∑l(W_(k,l))^2
#     print(reg * np.sum(W*W)) #debug
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
#     print('final loss: {0}'.format(loss))
#     print('final dW: {0}'.format(dW))

    return loss, dW


# In[47]:


def svm_loss_vectorized(X, y, W, reg = 0.0001):
    """
    A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside 
  this function)

  Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) #dW.shape = (num_total_pixels+1, num_classes)
    delta = 1.0
    scores = X.dot(W) # scores.shape = (len(X[0]), num_classes)
    correct_scores = scores[np.arange(X.shape[0]),y]
    num_classes = W.shape[1]
    num_samples = X.shape[0]
    
    # Implement a vectorized version of the structured SVM loss, storing 
    # the result in loss.   
    
    margins = np.maximum(0, ((scores.T - correct_scores).T) + delta)
    # margins.shape = (len(X[0]), num_classes), same as scores.shape
#     print('margins matrix before y: {0}'.format(margins))
    margins[np.arange(X.shape[0]),y] = 0
#     print('margins matrix: {0}'.format(margins)) #debug
    loss = np.sum(margins)/num_samples + reg * np.sum(W * W)
#     print('loss: {0}'.format(loss)) #debug
    margins[margins >0] = 1
    margins[np.arange(X.shape[0]),y] = - margins.sum(axis = 1)   
    dW = X.T.dot(margins)/num_samples + reg * W
#     print('dW: {0}'.format(dW))
    
    # Implement a vectorized version of the gradient for the structured 
    # SVM loss, storing the result in dW. 

    return loss, dW  


# ### next cell for debug only

# In[35]:


################################## Debug ###############################

# ### Load the raw CIFAR-10 data.
# cifar10_dir = '/Users/xiaoxiaoma/cifar-10-batches-py'

# # Cleaning up variables to prevent loading data multiple times (which \
# # may cause memory issue)
# try:
#     del X_train, y_train
#     del X_test, y_test
#     print('Clear previously loaded data.')
# except:
#     pass
# X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# mask_trial = 1
# X_trial = X_train[mask]
# y_trial = y_train[mask]

# X_trial = np.reshape(X_trial, (X_trial.shape[0], -1))
# X_trial -= mean_image
# X_trial = np.hstack([X_trial, np.ones((X_trial.shape[0], 1))])

# X_trial = X_trial[:5]
# y_trial = y_trial[:5]

# W_trial = init_W(X_trial, y_trial)

# loss, grad = svm_loss_naive(X_trial, y_trial, W_trial, 0.000005)
# loss_v, grad_v = svm_loss_vectorized(X_trial, y_trial, W_trial, 0.000005)

########################### Debug done ###########################


# ### test functionality

# In[49]:


W_dev = init_W(X_dev, y_dev)


# In[50]:


loss, grad = svm_loss_naive(X_dev, y_dev, W_dev, 0.000005)
print('loss: %f' % (loss, ))
print(grad.shape)
print('grad: {0}'.format(grad[:1, :3]))

loss_v, grad_v = svm_loss_vectorized(X_dev, y_dev, W_dev, 0.000005)
print('loss_v: %f' % (loss_v, ))
print(grad_v.shape)
print('grad_v: {0}'.format(grad_v[:1, :3]))


# In[52]:


# Numerically compute the gradient along several randomly chosen 
# dimensions, and compare them with your analytically computed gradient. 
# The numbers should match almost exactly along all dimensions.

from cs231n.gradient_check import grad_check_sparse
f = lambda w: svm_loss_naive(X_dev, y_dev, w, 0.0)[0]
grad_numerical = grad_check_sparse(f, W_dev, grad)


# In[53]:


# do the gradient check once again with regularization turned on
# you didn't forget the regularization gradient did you?

loss, grad = svm_loss_naive(X_dev, y_dev, W_dev, 5e1)
f = lambda w: svm_loss_naive(X_dev, y_dev, w, 5e1)[0]
grad_numerical = grad_check_sparse(f, W_dev, grad)
        


# In[54]:


# Next implement the function svm_loss_vectorized; for now only compute 
# the loss; we will implement the gradient in a moment.
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(X_dev, y_dev, W_dev,0.000005)
toc = time.time()
print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))

tic = time.time()
loss_vectorized, grad_vectorized = svm_loss_vectorized(X_dev, y_dev,                                                        W_dev, 0.000005)
toc = time.time()
print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# The losses should match but your vectorized implementation should be \
# much faster.
print('difference: %f' % (loss_naive - loss_vectorized))


# In[55]:


# Complete the implementation of svm_loss_vectorized, and compute the 
# gradient of the loss function in a vectorized way.

# The naive implementation and the vectorized implementation should match, 
# but the vectorized version should still be much faster.
tic = time.time()
_, grad_naive = svm_loss_naive(X_dev,y_dev,W_dev,0.000005)
toc = time.time()
print('Naive loss and gradient: computed in %fs' % (toc - tic))
print(grad_naive)

tic = time.time()
_, grad_vectorized = svm_loss_vectorized(X_dev, y_dev,                                          W_dev,0.000005)
toc = time.time()
print('Vectorized loss and gradient: computed in %fs' % (toc - tic))
print(grad_vectorized)

# The loss is a single number, so it is easy to compare the values 
# computed by the two implementations. The gradient on the other hand is 
# a matrix, so we use the Frobenius norm to compare them.
difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('difference: %f' % difference)


# In[81]:


# cd /Users/xiaoxiaomaDocuments/GitHub/machinelearningbasics/Stanfordcs231n


# In[82]:


# ls

