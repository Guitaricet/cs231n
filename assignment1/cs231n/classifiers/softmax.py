import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  f = X @ W
  for i in range(num_train):
    logC = np.max(f[i])  # numerical stability constant
    exp_f = np.exp(f[i] + logC)
    norm = np.sum(exp_f)
    loss -= np.log(exp_f[y[i]] / norm)
    for j in range(num_classes):
      softmax_output = exp_f[j] / norm
      if j != y[i]:
        dW[:, j] += softmax_output * X[i]
      else:
        dW[:, j] += (softmax_output - 1) * X[i]

  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += reg * 2*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  f = X @ W
  exp_f = np.exp(f + np.max(f, axis=1, keepdims=True))
  norm = np.sum(exp_f, axis=1)
  loss -= np.sum(np.log(exp_f[range(num_train), y] / norm))
  
  dW += X.T @ (exp_f / norm.reshape(-1, 1))
  tmp = np.zeros((num_train, num_classes))
  tmp[range(num_train), y] = 1.
  dW -= X.T @ tmp

  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += reg * 2*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

