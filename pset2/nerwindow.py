from numpy import *
from nn.base import NNBase
from nn.math import softmax, make_onehot
from misc import random_weight_matrix


##
# Evaluation code; do not change this
##
from sklearn import metrics
def full_report(y_true, y_pred, tagnames):
    cr = metrics.classification_report(y_true, y_pred,
                                       target_names=tagnames)
    print cr

def eval_performance(y_true, y_pred, tagnames):
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    print "=== Performance (omitting 'O' class) ==="
    print "Mean precision:  %.02f%%" % (100*sum(pre[1:] * support[1:])/sum(support[1:]))
    print "Mean recall:     %.02f%%" % (100*sum(rec[1:] * support[1:])/sum(support[1:]))
    print "Mean F1:         %.02f%%" % (100*sum(f1[1:] * support[1:])/sum(support[1:]))
    
##
# Implement this!
##
"""

In order to implement a model, you need to subclass NNBase, then implement the following methods:

__init__() (initialize parameters and hyperparameters)
_acc_grads() (compute and accumulate gradients)
compute_loss() (compute loss for a training example)
predict(), predict_proba(), or other prediction method (for evaluation)

NNBase provides you with a few others that will be helpful:

grad_check() (run a gradient check - calls _acc_grads and compute_loss)
train_sgd() (run SGD training; more on this later)

"""
class WindowMLP(NNBase):
    """Single hidden layer, plus representation learning."""

    def __init__(self, wv, windowsize=3,
                 dims=[None, 100, 5],
                 reg=0.001, alpha=0.01, rseed=10):
        """
        Initialize classifier model.

        Arguments:
        wv : initial word vectors (array |V| x n) n=50
            note that this is the transpose of the n x |V| matrix L
            described in the handout; you'll want to keep it in
            this |V| x n form for efficiency reasons, since numpy
            stores matrix rows continguously.
        windowsize : int, size of context window
        dims : dimensions of [input, hidden, output]
            input dimension can be computed from wv.shape
        reg : regularization strength (lambda)
        alpha : default learning rate
        rseed : random initialization seed
        """

        # Set regularization
        self.lreg = float(reg)
        self.alpha = alpha # default training rate
        self.nclass = dims[2]
        
        # input dimension, wv.shape is the dimension of each word vector representation
        dims[0] = windowsize * wv.shape[1] # 50*3
        param_dims = dict(W=(dims[1], dims[0]), # 100*150
                          b1=(dims[1]),
                          U=(dims[2], dims[1]),
                          b2=(dims[2]))
        param_dims_sparse = dict(L=wv.shape) # L.shape = (|V|*50)

        # initialize parameters: don't change this line
        NNBase.__init__(self, param_dims, param_dims_sparse)

        random.seed(rseed) # be sure to seed this for repeatability!

        self.params.W = random_weight_matrix(*self.params.W.shape) # 100*150
        self.params.U = random_weight_matrix(*self.params.U.shape) # 5*100
        #self.params.b1 = zeros((dims[1],))  # 100*1
        #self.params.b2 = zeros((self.nclass,)) # 5*1
        
        self.sparams.L = wv.copy()        


    def _acc_grads(self, window, label):
        """
        Accumulate gradients, given a training point
        (window, label) of the format

        window = [x_{i-1} x_{i} x_{i+1}] # three ints
        label = {0,1,2,3,4} # single int, gives class

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.U += (your gradient dJ/dU)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # this adds an update for that index
        """
        #### YOUR CODE HERE ####
        # Forward propagation
        #words = [self.sparams.L[window[0]], self.sparams.L[window[1]], self.sparams.L[window[2]]]
        #x = reshape(words, self.sparams.L.shape[1] *3) # 3n row vector 
        x = self.sparams.L[window, :].flatten()
        h = tanh(self.params.W.dot(x) + self.params.b1) # 100*1
        yhat = softmax(self.params.U.dot(h) + self.params.b2) # 5*1
        
        # Compute gradients w.r.t cross-entropy loss
        # Backpropagation
        y = make_onehot(label, len(yhat))
        delta = yhat - y
        
        # dJ/dU, dJ/db2
        self.grads.U += (outer(delta, h) + self.lreg * self.params.U)
        self.grads.b2 += delta
        
        # dJ/dW, dJ/db1
        delta2 = multiply((1 - square(h)), self.params.U.T.dot(delta))
        self.grads.W += (outer(delta2, x) + self.lreg * self.params.W)
        self.grads.b1 += delta2

        
        # dJ/dL, sparse grad update
        dJ_dL = self.params.W.T.dot(delta2).reshape(len(window), self.sparams.L.shape[1])
                
        #for i, w in enumerate(window):
        #    self.sgrads.L[w] = dJ_dL[i]
        for k in range(len(window)):
            self.sgrads.L[window[k]] = dJ_dL[k]


        #### END YOUR CODE ####


    def predict_proba(self, windows):
        """
        Predict class probabilities.

        Should return a matrix P of probabilities,
        with each row corresponding to a row of X.

        windows = array (n x windowsize),
            each row is a window of indices
        """
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]
        
        P = empty((len(windows), self.nclass))
        #### YOUR CODE HERE ####
        for i, row in enumerate(windows):

            x = self.sparams.L[row, :].flatten()
            #words = [self.sparams.L[row[0]], self.sparams.L[row[1]], self.sparams.L[row[2]]]
            #x = reshape(words, self.sparams.L.shape[1] *3) # 3n row vector
            h = tanh(self.params.W.dot(x) + self.params.b1)
        
            p = softmax(self.params.U.dot(h) + self.params.b2)
            P[i, :] = p
        #### END YOUR CODE ####

        return P # rows are output for each input


    def predict(self, windows):
        """
        Predict most likely class.
        Returns a list of predicted class indices;
        input is same as to predict_proba
        """

        P = self.predict_proba(windows)
        
        return argmax(P, axis=1) # list of predicted classes


    def compute_loss(self, windows, labels):
        """
        Compute the loss for a given dataset.
        windows = same as for predict_proba
        labels = list of class labels, for each row of windows
        """
        
        #### YOUR CODE HERE ####

        p = self.predict_proba(windows)
        J = -sum(log(choose(labels, p.T)))
        Jreg = self.lreg / 2.0 * (sum(self.params.W**2.0) + sum(self.params.U**2.0))
        J += Jreg
        #### END YOUR CODE ####
        return J
