import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Initializes a linear (fully connected) layer. 
        TODO: Initialize weights and biases.
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        """
        # Initialize weights and biases with the correct shapes.
        self.params = {'weight': None, 'bias': None}
        self.grads = {'weight': None, 'bias': None}
        self.params['weight'] = np.random.randn(in_features, out_features) * np.sqrt(1. / in_features)
        self.params['bias'] = np.zeros((1, out_features))

    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        TODO: Implement the forward pass.
        """
        self.x = x
        output = np.dot(x, self.params['weight']) + self.params['bias']
        return output

    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        TODO: Implement the backward pass.
        """
        self.grads["weight"] = np.dot(self.x.T, dout)
        self.grads["bias"] = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.grads["weight"].T)

class ReLU(object):
    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        TODO: Implement the forward pass.
        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        TODO: Implement the backward pass.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        """
        dout[self.mask] = 0
        dx = dout
        return dx

class SoftMax(object):
    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        TODO: Implement the forward pass using the Max Trick for numerical stability.
        """
        x = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(x)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.probs = probs
        return probs

    def backward(self, dout):
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        TODO: Keep this in mind when implementing CrossEntropy's backward method.
        """
        dx = np.zeros_like(dout)
        for i, (prob, d) in enumerate(zip(self.probs, dout)):
            J = np.diag(prob) - np.outer(prob, prob)
            dx[i] = np.dot(J, d)
        return dx

class CrossEntropy(object):
    def forward(self, x, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        TODO: Implement the forward pass.
        """
        loss = -np.sum(y * np.log(x + 1e-9)) / x.shape[0]  # Add a small value to prevent log(0).
        return loss

    def backward(self, x, y):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        TODO: Implement the backward pass.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        dx = (x - y) / y.shape[0]
        return dx
