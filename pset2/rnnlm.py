from numpy import *
import itertools
import time
import sys

# Import NN utils
from nn.base import NNBase
from nn.math import softmax, sigmoid, make_onehot
from nn.math import MultinomialSampler, multinomial_sample
from misc import random_weight_matrix


class RNNLM(NNBase):
    """
    Implements an RNN language model of the form:
    h(t) = sigmoid(H * h(t-1) + L[x(t)])
    y(t) = softmax(U * h(t))
    where y(t) predicts the next word in the sequence

    U = |V| * dim(h) as output vectors
    L = |V| * dim(h) as input vectors

    You should initialize each U[i,j] and L[i,j]
    as Gaussian noise with mean 0 and variance 0.1

    Arguments:
        L0 : initial input word vectors
        U0 : initial output word vectors
        alpha : default learning rate
        bptt : number of backprop timesteps
    """
    
    def __init__(self, L0, U0=None,
                 alpha=0.005, rseed=10, bptt=1):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        
        param_dims = dict(H = (self.hdim, self.hdim),
                          U = L0.shape)
        # note that only L gets sparse updates
        param_dims_sparse = dict(L = L0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)


        # Initialize word vectors
        # either copy the passed L0 and U0 (and initialize in your notebook)
        # or initialize with gaussian noise here
        self.sparams.L = random.normal(0, 0.1, L0.shape)
        self.params.U = random.normal(0, 0.1, U0.shape)
        
        # Initialize H matrix, as with W and U in part 1
        self.params.H = random_weight_matrix(self.hdim, self.hdim)
        self.bptt = bptt
        self.alpha = alpha   
    
        
    def _acc_grads(self, xs, ys):
    
        """
        Accumulate gradients, given a pair of training sequences:
        xs = [<indices>] # input words
        ys = [<indices>] # output words (to predict)

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.H += (your gradient dJ/dH)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # update row, matrix L: |V| * dim(h)

        Per the handout, you should:
            - make predictions by running forward in time
                through the entire input sequence
            - for *each* output word in ys, compute the
                gradients with respect to the cross-entropy
                loss for that output word
            - run backpropagation-through-time for self.bptt
                timesteps, storing grads in self.grads (for H, U)
                and self.sgrads (for L)

        You'll want to store your predictions \hat{y}(t)
        and the hidden layer values h(t) as you run forward,
        so that you can access them during backpropagation.

        At time 0, you should initialize the hidden layer to
        be a vector of zeros.
        """
    
        # Expect xs as list of indices
        ns = len(xs)

        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros) # compact h(t)s to a matrix
        hs = zeros((ns+1, self.hdim))
        # predicted probas
        ps = zeros((ns, self.vdim))
        

        # Forward propagation
        
        for i in range(ns):
            z1 = self.params.H.dot(hs[i-1]) + self.sparams.L[xs[i]]
            hs[i] = sigmoid(z1)
            z2 = self.params.U.dot(hs[i])
            ps[i] = softmax(z2)
   
        ps_copy = ps.copy()
        ps_copy[arange(len(ys)), ys] -= 1. # Matrix, each row is a prob distribution for predicting certain word.
        yhat_y = ps_copy
        
        mean_grads_H = zeros_like(self.params.H)
        mean_grads_U = zeros_like(self.params.U)

        # Backward propagation through time
        for t in reversed(range(ns)): 

            # Start from the latest step, e.g: 4, 3, 2, 1, 0
            mean_grads_U += outer(yhat_y[t], hs[t])
            delta = (self.params.U.T.dot(yhat_y[t])) * (hs[t] * (1-hs[t]))

            for s in range(min(t, self.bptt)+1):
            #for s in range(max(0, t-self.bptt), t+1):
                mean_grads_H += outer(delta, hs[t-s-1])
                self.sgrads.L[xs[t-s]] = delta 
                delta = self.params.H.T.dot(delta) * (hs[t-s-1] * (1-hs[t-s-1]))
        
        self.grads.H += mean_grads_H
        self.grads.U += mean_grads_U
          

    def grad_check(self, x, y, outfd=sys.stderr, **kwargs):
        """
        Wrapper for gradient check on RNNs;
        ensures that backprop-through-time is run to completion,
        computing the full gradient for the loss as summed over
        the input sequence and predictions.

        Do not modify this function!
        """
        bptt_old = self.bptt
        self.bptt = len(y)
        print >> outfd, "NOTE: temporarily setting self.bptt = len(y) = %d to compute true gradient." % self.bptt
        NNBase.grad_check(self, x, y, outfd=outfd, **kwargs)
        self.bptt = bptt_old
        print >> outfd, "Reset self.bptt = %d" % self.bptt


    def compute_seq_loss(self, xs, ys):
        """
        Compute the total cross-entropy loss
        for an input sequence xs and output
        sequence (labels) ys.

        You should run the RNN forward,
        compute cross-entropy loss at each timestep,
        and return the sum of the point losses.
        """


       
        ns = len(xs)
        hs = zeros((ns+1, self.hdim))
        ps = zeros((ns, self.vdim))
        
        for i in range(ns):
            z1 = self.params.H.dot(hs[i-1]) + self.sparams.L[xs[i]]
            hs[i] = sigmoid(z1)
            z2 = self.params.U.dot(hs[i])
            ps[i] = softmax(z2)        

        J = sum(-log(ps[range(len(ps)), ys]))

        return J


    def compute_loss(self, X, Y):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)

        Do not modify this function!
        """
        if not isinstance(X[0], ndarray): # single example
            return self.compute_seq_loss(X, Y)
        else: # multiple examples
            return sum([self.compute_seq_loss(xs,ys)
                       for xs,ys in itertools.izip(X, Y)])

    def compute_mean_loss(self, X, Y):
        """
        Normalize loss by total number of points.

        Do not modify this function!
        """
        J = self.compute_loss(X, Y)
        ntot = sum(map(len,Y))
        return J / float(ntot)


    def generate_sequence(self, init, end, maxlen=100):
        """
        Generate a sequence from the language model,
        by running the RNN forward and selecting,
        at each timestep, a random word from the
        a word from the emitted probability distribution.

        The MultinomialSampler class (in nn.math) may be helpful
        here for sampling a word. Use as:

            y = multinomial_sample(p)

        to sample an index y from the vector of probabilities p.


        Arguments:
            init = index of start word (word_to_num['<s>'])
            end = index of end word (word_to_num['</s>'])
            maxlen = maximum length to generate

        Returns:
            ys = sequence of indices
            J = total cross-entropy loss of generated sequence
        """

        J = 0 # total loss
        ys = [init] # emitted sequence
        
        hs = zeros((maxlen+1, self.hdim))
        ps = zeros((maxlen, self.vdim))
        
        for w in range(maxlen):
            z1 = self.params.H.dot(hs[w-1]) + self.sparams.L[ys[w]]
            hs[w] = sigmoid(z1)
            z2 = self.params.U.dot(hs[w])
            ps = softmax(z2)            
            y = multinomial_sample(ps)
            ys.append(y)
            J += -log(ps[y])
            if y == end:
                break
                
        return ys, J



class ExtraCreditRNNLM(RNNLM):
    """
    Implements an improved RNN language model,
    for better speed and/or performance.

    We're not going to place any constraints on you
    for this part, but we do recommend that you still
    use the starter code (NNBase) framework that
    you've been using for the NER and RNNLM models.
    """

    def __init__(self, *args, **kwargs):
        #### YOUR CODE HERE ####
        raise NotImplementedError("__init__() not yet implemented.")
        #### END YOUR CODE ####

    def _acc_grads(self, xs, ys):
        #### YOUR CODE HERE ####
        raise NotImplementedError("_acc_grads() not yet implemented.")
        #### END YOUR CODE ####

    def compute_seq_loss(self, xs, ys):
        #### YOUR CODE HERE ####
        raise NotImplementedError("compute_seq_loss() not yet implemented.")
        #### END YOUR CODE ####

    def generate_sequence(self, init, end, maxlen=100):
        #### YOUR CODE HERE ####
        raise NotImplementedError("generate_sequence() not yet implemented.")
        #### END YOUR CODE ####