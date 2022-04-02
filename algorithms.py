import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, log_loss


def neg_log_lik(y_pred, y_true, w, mu=0):
    y_pred = y_pred.copy()
    y_pred[y_pred == 0] = 1e-14
    y_pred[y_pred == 1] = 1 - 1e-14
    return -(y_true@np.log(y_pred)) - (1-y_true)@np.log(1-y_pred) + mu*np.sum(np.square(w))

def gradient(X, y_true, y_pred, w, mu=0):
    return ((y_pred - y_true)@X + 2*mu*w)

def sigmoid(wTx):
    return 1/(1+np.exp(-wTx))

def StochasticGradient(
    X,
    y, 
    mu, 
    lr, 
    batch_size,
    max_epoch, 
    random_state=3136,
):
    N = X.shape[0]
    m = X.shape[1]
    idx = list(range(N))
    
    n_iter = (N-1)//batch_size + 1
    np.random.seed(random_state)
    w = np.random.rand(m)
    
    y_pred = sigmoid(X @ w)
    print(f"Starting, Loss: {neg_log_lik(y_pred, y, w, mu)}")
    
    lr0 = lr
    for epoch in range(max_epoch):
        
        print("="*20, "EPOCH:", epoch, "="*20)
        # Shuffle points at each iteration
        np.random.shuffle(idx)
        X_shuffled = X[idx, :]
        y_shuffled = y[idx]
        
        for i in range(n_iter):
            # Get batch
            X_batch = X_shuffled[batch_size*i:batch_size*(i+1),:]
            y_batch = y_shuffled[batch_size*i:batch_size*(i+1)]
            y_pred = sigmoid(X_batch @ w)
            # Get gradient estimate
            g = gradient(X_batch, y_batch, y_pred, w, mu)
            # Update weights
            w = w - lr*g

        lr = lr0*(1/(epoch+1))
        # Report total loss
        y_pred = sigmoid(X @ w)
        g = gradient(X, y, y_pred, w, mu)
        print(f"Train Balanced Acc: {balanced_accuracy_score(y, y_pred>0.5)}")
        print(f"Length gradient: {np.dot(g,g)**0.5}")
        print(f"Loss: {neg_log_lik(y_pred, y, w, mu)}")
    
    return w


def MomentumStochasticGradient(
    X,
    y, 
    mu, 
    lr,
    moment,
    batch_size,
    max_epoch, 
    random_state=3136,
):
    N = X.shape[0]
    m = X.shape[1]
    idx = list(range(N))
    
    n_iter = (N-1)//batch_size + 1
    np.random.seed(random_state)
    w = np.random.rand(m)
    
    y_pred = sigmoid(X @ w)
    print(f"Starting, Loss: {neg_log_lik(y_pred, y, w, mu)}")
    
    lr0 = lr
    v = np.zeros(m)
    for epoch in range(max_epoch):
        
        print("="*20, "EPOCH:", epoch, "="*20)
        # Shuffle points at each iteration
        np.random.shuffle(idx)
        X_shuffled = X[idx, :]
        y_shuffled = y[idx]
        
        for i in range(n_iter):
            # Get batch
            X_batch = X_shuffled[batch_size*i:batch_size*(i+1),:]
            y_batch = y_shuffled[batch_size*i:batch_size*(i+1)]
            y_pred = sigmoid(X_batch @ w)
            # Get gradient estimate
            g = gradient(X_batch, y_batch, y_pred, w, mu)
            # Update weights
            v = moment*v + lr*g
            w = w - v

        lr = lr0*(1/(epoch+1))
        # Report total loss
        y_pred = sigmoid(X @ w)
        g = gradient(X, y, y_pred, w, mu)
        print(f"Train Balanced Acc: {balanced_accuracy_score(y, y_pred>0.5)}")
        print(f"Length gradient: {np.dot(g,g)**0.5}")
        print(f"Loss: {neg_log_lik(y_pred, y, w, mu)}")
    
    return w

def AdaGrad(
    X,
    y, 
    mu, 
    lr,
    batch_size,
    max_epoch, 
    random_state=3136,
):
    N = X.shape[0]
    m = X.shape[1]
    idx = list(range(N))
    
    n_iter = (N-1)//batch_size + 1
    np.random.seed(random_state)
    w = np.random.rand(m)
    
    y_pred = sigmoid(X @ w)
    print(f"Starting, Loss: {neg_log_lik(y_pred, y, w, mu)}")
    
    lr0 = lr
    sq_grad = np.zeros(m)
    for epoch in range(max_epoch):
        
        print("="*20, "EPOCH:", epoch, "="*20)
        # Shuffle points at each iteration
        np.random.shuffle(idx)
        X_shuffled = X[idx, :]
        y_shuffled = y[idx]
        
        for i in range(n_iter):
            # Get batch
            X_batch = X_shuffled[batch_size*i:batch_size*(i+1),:]
            y_batch = y_shuffled[batch_size*i:batch_size*(i+1)]
            y_pred = sigmoid(X_batch @ w)
            # Get gradient estimate
            g = gradient(X_batch, y_batch, y_pred, w, mu)
            sq_grad += g**2
            update = g/np.sqrt(sq_grad + 1e-10)
            # Update weights
            w = w - lr*update

        lr = lr0*(1/(epoch+1))
        # Report total loss
        y_pred = sigmoid(X @ w)
        g = gradient(X, y, y_pred, w, mu)
        print(f"Train Balanced Acc: {balanced_accuracy_score(y, y_pred>0.5)}")
        print(f"Length gradient: {np.dot(g,g)**0.5}")
        print(f"Loss: {neg_log_lik(y_pred, y, w, mu)}")
    
    return w

def HingeLossl2(C, w, w0, X, y):
    loss = (1/(2*C))*np.dot(w.T,w) + np.sum(np.maximum(0,1-(X@w+w0)*y))
    return loss.squeeze()

def subgradient(C, w, w0, X, y):
    margins = 1-(X@w+w0)*y
    margins = margins.squeeze()
    line0_grad = np.zeros(X.shape)
    hyperplane_grad = (-X*y)
    
    # since line0 grad is a valid option for margin == 0 subgradient
    grad = (
        (1/(2*C))*w  
        + hyperplane_grad[margins>0].sum(axis=0).reshape(-1,1) 
        + line0_grad[margins<=0].sum(axis=0).reshape(-1,1)
    )
    grad0 = np.sum(-y[margins>0])
    grad = grad/np.sqrt(np.sum(grad**2))
    # Return normalized subgradient
    
    return grad0, grad

def StochasticSubGradient(
    X,
    y, 
    C, 
    lr,
    batch_size,
    max_epoch, 
    random_state=3136,
):
    N = X.shape[0]
    m = X.shape[1]
    idx = list(range(N))
    
    n_iter = (N-1)//batch_size + 1
    np.random.seed(random_state)
    w = np.random.rand(m,1)
    w0 = np.random.rand(1,1)
    
    bacc_y = y.copy()
    bacc_y[bacc_y == -1] = 0
    
    print(f"Starting, Loss: {HingeLossl2(C, w, w0, X, y)}")
    
    lr0 = lr
    for epoch in range(max_epoch):
        print("="*20, "EPOCH:", epoch, "="*20)
        # Shuffle points at each iteration
        np.random.shuffle(idx)
        X_shuffled = X[idx, :]
        y_shuffled = y[idx, :]
        
        for i in range(n_iter):
            # Get batch
            X_batch = X_shuffled[batch_size*i:batch_size*(i+1),:]
            y_batch = y_shuffled[batch_size*i:batch_size*(i+1), :]
            # Get gradient estimate
            g0, g = subgradient(C, w, w0, X_batch, y_batch)
            # Update weights
            w = w - lr*g
            w0 = w0 - lr*g0

        lr = lr0*(1/(epoch+1))
        # Report total loss
        y_pred = (X@w + w0) > 0
        print(np.unique(y_pred, return_counts=True))
        print(f"Train Balanced Acc: {balanced_accuracy_score(bacc_y, y_pred)}")
        print(f"Loss: {HingeLossl2(C, w, w0, X, y)}")
    
    return w0 ,w

def MomentumSubGradient(
    X,
    y, 
    C, 
    lr,
    moment,
    batch_size,
    max_epoch, 
    random_state=3136,
):
    N = X.shape[0]
    m = X.shape[1]
    idx = list(range(N))
    
    n_iter = (N-1)//batch_size + 1
    np.random.seed(random_state)
    w = np.random.rand(m,1)
    w0 = np.random.rand(1,1)
    
    bacc_y = y.copy()
    bacc_y[bacc_y == -1] = 0
    
    print(f"Starting, Loss: {HingeLossl2(C, w, w0, X, y)}")
    
    lr0 = lr
    v = np.zeros((m,1))
    v0 = 0
    for epoch in range(max_epoch):
        print("="*20, "EPOCH:", epoch, "="*20)
        # Shuffle points at each iteration
        np.random.shuffle(idx)
        X_shuffled = X[idx, :]
        y_shuffled = y[idx, :]
        
        for i in range(n_iter):
            # Get batch
            X_batch = X_shuffled[batch_size*i:batch_size*(i+1),:]
            y_batch = y_shuffled[batch_size*i:batch_size*(i+1), :]
            # Get gradient estimate
            g0, g = subgradient(C, w, w0, X_batch, y_batch)
            # Update weights
            v = moment*v + lr*g
            v0 = moment*v0 + lr*g0
            w = w - v
            w0 = w0 - v0

        lr = lr0*(1/(epoch+1))
        # Report total loss
        y_pred = (X@w + w0) > 0
        print(np.unique(y_pred, return_counts=True))
        print(f"Train Balanced Acc: {balanced_accuracy_score(bacc_y, y_pred)}")
        print(f"Loss: {HingeLossl2(C, w, w0, X, y)}")
    
    return w0 ,w


def AdaSubGradient(
    X,
    y, 
    C, 
    lr,
    batch_size,
    max_epoch, 
    random_state=3136,
):
    N = X.shape[0]
    m = X.shape[1]
    idx = list(range(N))
    
    n_iter = (N-1)//batch_size + 1
    np.random.seed(random_state)
    w = np.random.rand(m,1)
    w0 = np.random.rand(1,1)
    
    bacc_y = y.copy()
    bacc_y[bacc_y == -1] = 0
    
    print(f"Starting, Loss: {HingeLossl2(C, w, w0, X, y)}")
    
    lr0 = lr
    sq_grad = np.zeros((m,1))
    sq_grad0 = 0
    for epoch in range(max_epoch):
        print("="*20, "EPOCH:", epoch, "="*20)
        # Shuffle points at each iteration
        np.random.shuffle(idx)
        X_shuffled = X[idx, :]
        y_shuffled = y[idx, :]
        
        for i in range(n_iter):
            # Get batch
            X_batch = X_shuffled[batch_size*i:batch_size*(i+1),:]
            y_batch = y_shuffled[batch_size*i:batch_size*(i+1), :]
            # Get gradient estimate
            g0, g = subgradient(C, w, w0, X_batch, y_batch)
            # Update weights
            sq_grad += g**2
            sq_grad0 += g0**2
            update = g/np.sqrt(sq_grad + 1e-10)
            update0 = g0/np.sqrt(sq_grad0 + 1e-10)
            w = w - lr*update
            w0 = w0 - lr*update0

        lr = lr0*(1/(epoch+1))
        # Report total loss
        y_pred = (X@w + w0) > 0
        print(np.unique(y_pred, return_counts=True))
        print(f"Train Balanced Acc: {balanced_accuracy_score(bacc_y, y_pred)}")
        print(f"Loss: {HingeLossl2(C, w, w0, X, y)}")
    
    return w0 ,w