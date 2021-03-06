
# coding: utf-8

# In[42]:


# cd Documents/GitHub/machinelearningbasics/Stanfordcs231n/


# In[52]:


# ls


# In[64]:


from __future__ import print_function
import numpy as np

import import_ipynb
import LossF_SVM
import LossF_Softmax


# In[65]:


class LinearClassifier(object):
    def __init__(self):
        self.W = None
        
    def train(self, X, y, learning_rate = 1e-7, reg = 1e-5,               num_iters = 1500, batch_size = 200, verbose = False):
        """
        Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; 
      there are N training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; 
      y[i] = c means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at 
      each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training
    iteration.
        """

        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            # naive initialization
            self.W = 0.001 * np.random.randn(dim, num_classes)
#         print('original W: {}'.format(self.W[:5]))
        
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None
    
    #################################################################
      # Sample batch_size elements from the training data and their 
      # corresponding labels to use in this round of gradient descent.
      # Store the data in X_batch and their corresponding labels in 
      # y_batch; after sampling X_batch should have shape (dim, 
      # batch_size) and y_batch should have shape (batch_size,) 
        
      # Hint: Use np.random.choice to generate indices. Sampling with 
      # replacement is faster than sampling without replacement.    
      #############################################################
    ##------------------- written by me start ---------------------##    

            batch_indices = np.random.choice(num_train, batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
  
    ##--------------------- written by me end ---------------------##   
    
    # evaluate loss and gradient
            loss, dW = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)  
    
    ##############################################################
      # Update the weights using the gradient and the learning rate.    
      ##############################################################
    ##------------------- written by me start ---------------------##        
            self.W += - learning_rate * dW 
            # perform parameter update
            
    ##--------------------- written by me end ---------------------##          

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
#         print('first 5 W = {}'.format(self.W[:5]))
        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict 
    labels for data points.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; 
      there are N training samples each of dimension D.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 
      1-dimensional array of length N, and each element is an integer 
      giving the predicted class.
        """

        y_pred = np.zeros(X.shape[0])
    #################################################################
    # TODO:                                                    
    # Implement this method. Store the predicted labels in y_pred.      
    #################################################################
    
    ##------------------- written by me start ---------------------##    
#         print('first 5 W = {}'.format(self.W[:5]))
        scores = X.dot(self.W)
#         print('scores.shape = {}'.format(scores.shape))
        y_pred = np.argmax(scores, axis = 1)
    ##--------------------- written by me end ---------------------##  
    
        return y_pred
  
    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch 
      of N data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the 
      minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
        """
        pass    
   


# In[66]:


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(X_batch, y_batch, self.W, reg)


# In[67]:


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(X_batch, y_batch, self.W, reg)


# In[57]:


import time


# In[55]:


print('imported')

