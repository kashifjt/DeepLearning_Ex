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
    D, C = np.shape(W)
    N = len(y)
    score = np.zeros((500, 10), dtype=np.float32)
    for indN in range(0, N):
        for indC in range(0, C):
            score[indN, indC] = sum(X[indN]*W[:, indC])
    score -= np.max(score)
    prob_score = np.zeros_like(score)  # 500, 10
    for indN in range(0, N):
        for indC in range(0, C):
            prob_score[indN, indC] = np.exp(score[indN, indC])
    norm_prob_score = np.zeros_like(score)
    for indN in range(0, N):
        for indC in range(0, C):
            norm_prob_score[indN, indC] = prob_score[indN, indC] / sum(prob_score[indN])
    for indN in range(0, N):
        loss = np.log(norm_prob_score[indN, y[indN]])
    loss = ((-1/N) * loss) + ((reg/2) * np.sum(W * W))
    upd_prob_score = np.zeros_like(norm_prob_score)
    for indN in range(0, N):
        for indC in range(0, C):
            if y[indN] != indC:
                upd_prob_score[indN, indC] = 0 - norm_prob_score[indN, indC]
            else:
                upd_prob_score[indN, indC] = 1 - norm_prob_score[indN, indC]
    for indD in range(0, D):
        for indC in range(0, C):
            dW[indD, indC] = np.sum(X.T[indD] * upd_prob_score[:, indC])
    dW = ((-1/N) * dW) + (reg * W)
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
    D, C = np.shape(W)
    N = len(y)
    score = np.dot(X, W)
    score -= np.max(score)
    norm_prob_score = np.exp(score) / np.sum(np.exp(score), axis=1)[np.newaxis].T
    for indN in range(0, N):
        loss = np.log(norm_prob_score[indN, y[indN]])
    loss = ((-1/N) * loss) + ((reg/2) * np.sum(W * W))
    upd_prob_score = np.zeros_like(norm_prob_score)
    for indN in range(0, N):
        for indC in range(0, C):
            if y[indN] != indC:
                upd_prob_score[indN, indC] = 0 - norm_prob_score[indN, indC]
            else:
                upd_prob_score[indN, indC] = 1 - norm_prob_score[indN, indC]
    dW = ((-1/N) * np.dot(X.T, upd_prob_score)) + (reg * W)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
