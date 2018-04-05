"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data) #if there is test_data, get the number of testing data points
        n = len(training_data) #get the number of training data points
        for j in range(epochs): #for each element j in the range of the number of epochs
            random.shuffle(training_data) #shuffle the training data
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] #third argument in xrange specifies a step, slice data into mini_batches
            #if mini_batch_size goes over the limit, a list of [k:n] will be used at the end
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) #calls an update function to adjust weights and bias (the call to actually do the machine learning bit)

            """
            Formatted output for user/developer to see what pass (epoch) through the training data
            we are currently on. If there is test data, there will also be a performance metric
            provided in addition to the epoch number.
            """
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)


    """
    Algorithm starts getting less intuitive here. The next two functions detail the process
    in which a neural network will "learn" via an error function and backpropagation.
    """
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        """
        This part of the algorithm actually updates the weights and biases using the values
        we got from the backpropagation algorithm. The eta/len(minibatch) is a regularization parameter
        that speeds or slows down the learning process but ignore that for now (you can assume that it evaluates to 1).
        """
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations (sigmoid(z) where z = w*x+b), layer by layer
        zs = [] # list to store all the z vectors (w*x+b), layer by layer. We use this to help with derivatives later.
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b #calculate value of z for each layer
            zs.append(z) #add z to the list of z vectors (append adds an element to the end of a list)
            activation = sigmoid(z) #calculate value (in this case a list) of sigmoid(z) for a layer
            activations.append(activation) #add this to a list containing each layer. Ex: activations[1] is the list of activations for the second layer.
        # backward pass
        """
        Now the neural network will take the partial derivative of an error function
        (output - target)^2 with respect to each bias and weight in the network in order
        to "step" towards a minimum that will eventually work across all of our training
        data.
        """
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        """
        Normally, if you apply the exponent rule and chain rule to this function (eta/(2*len(mini_batch)))(output-target)^2,
        with respect to the biases, you will get:

        (eta/(2*len(mini_batch)))*(2*(output - target) * d/db(output)).

        (eta/(2*len(mini_batch))) is just a constant so it does not change when taking the derivative.

        output = sigmoid(w*x+b)

        ->if you go back and look at the end of the update_mini_batch function. you can see that we use (eta/(len(mini_batch)))
        later, so we will ignore it for now. However, you can see that the 2 in the denominator is gone when we use it. You can also
        see in the line above and in the cost_derivative function that we are missing a 2. This is because the author of this code has
        canceled them out because they would normally reduce out.

        2*(output - target) * d/db(output) -> (output - target) * d/db(output) -cancel out 2 from 2 in the denominator of regularization parameter
        output = sigmoid(z), z = w*x+b

        Next Chain Rule step -> (output - target) * sigmoid_prime(z) * d/db(z)

        d/db(z) -> d/db(w*x+b) -> (0+1) = 1

        Full Chain Rule: (output - target) * sigmoid_prime(z) * 1, which is what we have in the line of code above

        Represents derivatives with respect to all the biases for the very last layer.
        """

        nabla_b[-1] = delta #Assign delta to bias derivatives because that is what delta is equivalent to.
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        """
        If we were to take the derivatives with respect to the weights, all that would change is d/dw(z)
        d/dw(z) = d/dw(w*x+b) = (x+0) = x
        Therefore, the derivative function looks like:

        (output - target) * sigmoid_prime(z) * d/dw(z)

        (output - target) * sigmoid_prime(z) * x

        ->delta * x

        x is the list of activations from the previous layer so we say that the list of weight derivatives
        for the output layer is delta * activations[-2] where activations[-2] are the activations in the second
        to last layer which would be the previous layer in this case.

        transpose is used to make the dimensions work out for matrix multiplication.

        """
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.

        """
        While the derivatives of the final output layer depend on our target value,
        the adjustments to layers before the last layer are computed by taking the
        derivative of the activations in the layer after. Because of this, computing the
        delta for the remaining layers is a repettitive process.

        The idea of this whole neural network algorithm is that we move forward through the network to make
        a prediction, and then we move backward through the network to adjust our parameter so that we make
        a better prediction on the next pass forward.
        """

        for l in range(2, self.num_layers):
        #l is kind of representing what layer we are on counting from the back, we already did the last layer so we can start at the second to last layer
            z = zs[-l] #get un-sigmoided activations for current layer.
            sp = sigmoid_prime(z) #get derivative of sigmoid(z), z = w*x+b
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
