# Deep learning
# Gradient Descent algorithm

import numpy as np


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.size = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1],sizes[1:])]


def sigmoid(z):
    return 1.0/(1.0+ np.exp(-z))


def derivation_sigmoid(z):
    return sigmoid(z) - sigmoid(z)*2


def feedforward(self, a):
    for w, b in zip(self.weights,self.biases):
        a = np.dot(w, a) + b
        return a


def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
    if test_data:
        n_test = len(test_data)

    n_train = len(training_data)
    for j in xrange(epochs):
        random.shuffle(training_data)
        mini_batches = [training_data[k:k+mini_batch_size,:] for k in xrange(0,n_train,mini_batch_size)]
        for mini_batch in mini_batches:
            self.updated_mini_batch(mini_batch,eta)
        if test_data:
            print('epoch {0}:{1} / {2}'.format(j, evaluate(test_data),n_test))
        else:
            print ('epoch {0} complete'.format(j))


def updated_mini_batch(self, mini_batch, eta):
    b = [np.zeros(w.shape) for w in self.weights]
    w = [np.zeros(b.shape) for b in self.biases]
    for x, y in mini_batch:
        nudge_b, nudge_w = backprop(x, y)
        updated_b = [b + nudge_b for b, nudge_b in zip(b,nudge_b)]
        updated_w = [w + nudge_w for w, nudge_w in zip(w,nudge_w)]
    self.weights = [ow - (eta/len(mini_batch))*nw for w, nw in zip(self.weights,updated_w)]
    self.biases = [ob- (eta/len(mini_batch))*nb for b, nb in zip(self.biases, updated_b)]


def backprop(self, x, y):
    # forward
    act_w = [np.zeros(w.shape) for w in self.weights]
    act_b = [np.zeros(w.shape) for b in self.biases]
    activation = x
    activations = [x]
    zs = []

    for w, b in zip(self.weights, self.biases):
        z = np.dot(act_w, x) + act_b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

    delta = self.loss_func(activations[-1], y)*self.derivation_sigmoid(zs[-1])
    nudge_b[-1] = delta
    nudge_w[-1] = np.dot(delta,activations[-2])

    for l in xrange(2, self.num_layers):
        delta = self.loss_func(activations[-l+1], y)*self.derivation_sigmoid(zs[-l])
        nudge_b[-l] = delta
        nudge_w[-l] = np.dot(delta,activations[-l-1])
        return (nudge_b, nudge_w)


def loss_func(act_output, y):
    return act_output - y

def evaluate(self,test_data):
    test_result = [(np.argmax(feedforward(x)),y) for x,y in test_data]
    return sum(int(x == y) for x,y in test_result)


