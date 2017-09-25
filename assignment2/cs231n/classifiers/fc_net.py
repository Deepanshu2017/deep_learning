from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        W1 = np.random.randn(input_dim, hidden_dim) * weight_scale
        W2 = np.random.randn(hidden_dim, num_classes) * weight_scale
        b1 = np.zeros(hidden_dim)
        b2 = np.zeros(num_classes)
        self.params['W1'] = W1
        self.params['W2'] = W2
        self.params['b1'] = b1
        self.params['b2'] = b2
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        X = X.reshape(X.shape[0], -1)

        hidden_layer = np.maximum(0, X.dot(W1) + b1)
        scores = hidden_layer.dot(W2) + b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        shift_scores = scores - np.max(scores, axis=1, keepdims=True)
        softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1, keepdims=True)
        loss = -np.sum(np.log(softmax_output[range(X.shape[0]), list(y)]))
        loss /= X.shape[0]
        reg =  self.reg
        loss = loss + (0.5 * reg * np.sum(W1 * W1))
        loss = loss + (0.5 * reg * np.sum(W2 * W2))

        N = X.shape[0]

        dscores = softmax_output
        dscores[range(N), list(y)] += -1.0
        dscores /= N

        dW2 = hidden_layer.T.dot(dscores)
        dW2 += reg * W2
        dB2 = np.sum(dscores, axis=0)

        dhidden_layer = dscores.dot(W2.T)
        dhidden_layer[hidden_layer <= 0] = 0

        dW1 = X.T.dot(dhidden_layer)
        dW1 += reg * W1
        dB1 = np.sum(dhidden_layer, axis=0)

        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['b1'] = dB1
        grads['b2'] = dB2

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################

        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params['W%d' % (i + 1)] = np.random.randn(dims[i], dims[i + 1]) * weight_scale
            self.params['b%d' % (i + 1)] = np.zeros(dims[i + 1])
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
            gammas = {'gamma' + str(i + 1): np.ones(dims[i + 1]) for i in range(self.num_layers - 1)}
            betas = {'beta' + str(i + 1): np.zeros(dims[i + 1]) for i in range(self.num_layers - 1)}

            self.params.update(betas)
            self.params.update(gammas)

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # W1, W2 = self.params['W1'], self.params['W2']
        # b1, b2 = self.params['b1'], self.params['b2']
        # hidden_layers = list()
        # hidden_layer = np.maximum(0, X.dot(W1) + b1)
        # scores = np.maximum(0, hidden_layer.dot(W2) + b2)

        layers = {0: X}
        cache = {}

        for i in  range(1, self.num_layers):
            a, fc_cache = affine_forward(layers[i - 1], self.params['W%d' % (i)], self.params['b%d' % (i)])
            if self.use_batchnorm:
                gamma = self.params['gamma' + str(i)]
                beta = self.params['beta' + str(i)]
                bn_param = self.bn_params[i - 1]
                a, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
                out, relu_cache = relu_forward(a)
                layers[i], cache[i] = out, (fc_cache, relu_cache, bn_cache)
            else:
                out, relu_cache = relu_forward(a)
                layers[i], cache[i] = out, (fc_cache, relu_cache)

        a, fc_cache = affine_forward(layers[self.num_layers - 1], self.params['W%d' % (self.num_layers)], self.params['b%d' % (self.num_layers)])
        layers[self.num_layers], cache[self.num_layers] = a, fc_cache

        scores = layers[self.num_layers]
        # print("Real score:", scores)
        # one = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
        # two = np.maximum(0, one.dot(self.params['W2']) + self.params['b2'])
        # scores = two.dot(self.params['W3']) + self.params['b3']
        # print("Score: ", scores)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        shift_scores = scores - np.max(scores, axis=1, keepdims=True)
        softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1, keepdims=True)
        loss = -np.sum(np.log(softmax_output[range(X.shape[0]), list(y)]))
        loss /= X.shape[0]
        reg = self.reg
        for i in range(1, self.num_layers + 1):
            loss += 0.5 * reg * np.sum(self.params['W%d' % i] * self.params['W%d' % i])

        dsoftmax = softmax_output
        dsoftmax[range(X.shape[0]), list(y)] -= 1.0
        dsoftmax /= X.shape[0]

        dx = {}
        dx[self.num_layers], grads['W%d' % self.num_layers], grads['b%d' % self.num_layers] = affine_backward(dsoftmax, cache[self.num_layers])
        grads['W%d' % self.num_layers] += reg * self.params['W%d' % self.num_layers]

        for i in reversed(range(1, self.num_layers)):
            if self.use_batchnorm:
                fc_cache, relu_cache, bn_cache = cache[i]
            else:
                fc_cache, relu_cache = cache[i]
            da = relu_backward(dx[i + 1], relu_cache)
            if self.use_batchnorm:
                da, dgamma, dbeta = batchnorm_backward(da, bn_cache)
                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta
                dx[i], grads['W%d' % i], grads['b%d' % i] = affine_backward(da, fc_cache)
                grads['W%d' % i] += reg * self.params['W%d' % i]
            else:
                dx[i], grads['W%d' % i], grads['b%d' % i] = affine_backward(da, fc_cache)
                grads['W%d' % i] += reg * self.params['W%d' % i]

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return loss, grads
