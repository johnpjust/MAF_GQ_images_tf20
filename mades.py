# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
import numpy as np
import numpy.random as rng

dtype = tf.float32

def create_degrees(n_inputs, n_hiddens, input_order, mode):
    """
    Generates a degree for each hidden and input unit. A unit with degree d can only receive input from units with
    degree less than d.
    :param n_inputs: the number of inputs
    :param n_hiddens: a list with the number of hidden units
    :param input_order: the order of the inputs; can be 'random', 'sequential', or an array of an explicit order
    :param mode: the strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
    :return: list of degrees
    """

    degrees = []

    # create degrees for inputs
    if isinstance(input_order, str):

        if input_order == 'random':
            degrees_0 = np.arange(1, n_inputs + 1)
            rng.shuffle(degrees_0)

        elif input_order == 'sequential':
            degrees_0 = np.arange(1, n_inputs + 1)

        else:
            raise ValueError('invalid input order')

    else:
        input_order = np.array(input_order)
        assert np.all(np.sort(input_order) == np.arange(1, n_inputs + 1)), 'invalid input order'
        degrees_0 = input_order
    degrees.append(degrees_0)

    # create degrees for hiddens
    if mode == 'random':
        for N in n_hiddens:
            min_prev_degree = min(np.min(degrees[-1]), n_inputs - 1)
            degrees_l = rng.randint(min_prev_degree, n_inputs, N)
            degrees.append(degrees_l)

    elif mode == 'sequential':
        for N in n_hiddens:
            degrees_l = np.arange(N) % max(1, n_inputs - 1) + min(1, n_inputs - 1)
            degrees.append(degrees_l)

    else:
        raise ValueError('invalid mode')

    return degrees

def create_masks(degrees):
    """
    Creates the binary masks that make the connectivity autoregressive.
    :param degrees: a list of degrees for every layer
    :return: list of all masks, as theano shared variables
    """

    Ms = []

    for l, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
        M = d0[:, np.newaxis] <= d1
        M = tf.constant(M, dtype=dtype, name='M' + str(l+1))
        Ms.append(M)

    Mmp = degrees[-1][:, np.newaxis] < degrees[0]
    Mmp = tf.constant(Mmp, dtype=dtype, name='Mmp')

    return Ms, Mmp

def create_weights(n_inputs, n_hiddens, n_comps):
    """
    Creates all learnable weight matrices and bias vectors.
    :param n_inputs: the number of inputs
    :param n_hiddens: a list with the number of hidden units
    :param n_comps: number of gaussian components
    :return: weights and biases, as tensorflow variables
    """

    Ws = []
    bs = []

    n_units = np.concatenate(([n_inputs], n_hiddens))

    for l, (N0, N1) in enumerate(zip(n_units[:-1], n_units[1:])):
        W = tf.Variable((rng.randn(N0, N1) / np.sqrt(N0 + 1)), dtype=dtype, name='W' + str(l+1))
        b = tf.Variable(np.zeros([1,N1]), dtype=dtype, name='b' + str(l+1))
        Ws.append(W)
        bs.append(b)

    if n_comps is None:

        Wm = tf.Variable((rng.randn(n_units[-1], n_inputs) / np.sqrt(n_units[-1] + 1)), dtype=dtype, name='Wm')
        Wp = tf.Variable((rng.randn(n_units[-1], n_inputs) / np.sqrt(n_units[-1] + 1)), dtype=dtype, name='Wp')
        bm = tf.Variable(np.zeros([1,n_inputs]), dtype=dtype, name='bm')
        bp = tf.Variable(np.zeros([1,n_inputs]), dtype=dtype, name='bp')

        return Ws, bs, Wm, bm, Wp, bp

    else:

        Wm = tf.Variable((rng.randn(n_units[-1], n_inputs, n_comps) / np.sqrt(n_units[-1] + 1)), dtype=dtype, name='Wm')
        Wp = tf.Variable((rng.randn(n_units[-1], n_inputs, n_comps) / np.sqrt(n_units[-1] + 1)), dtype=dtype, name='Wp')
        Wa = tf.Variable((rng.randn(n_units[-1], n_inputs, n_comps) / np.sqrt(n_units[-1] + 1)), dtype=dtype, name='Wa')
        bm = tf.Variable(rng.randn(n_inputs, n_comps), dtype=dtype, name='bm')
        bp = tf.Variable(rng.randn(n_inputs, n_comps), dtype=dtype, name='bp')
        ba = tf.Variable(rng.randn(n_inputs, n_comps), dtype=dtype, name='ba')

    return Ws, bs, Wm, bm, Wp, bp, Wa, ba

class GaussianMade:
    """
    Implements a Made, where each conditional probability is modelled by a single gaussian component.
    Reference: Germain et al., "MADE: Masked Autoencoder for Distribution Estimation", ICML, 2015.
    """

    def __init__(self, n_inputs, n_hiddens, act_fun, input_order='sequential', mode='sequential', input=None):
        """
        Constructor.
        :param n_inputs: number of inputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: tensorflow activation function
        :param input_order: order of inputs
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input: tensorflow placeholder to serve as input; if None, a new placeholder is created
        """

        # save input arguments
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.act_fun = act_fun
        self.mode = mode

        # create network's parameters
        degrees = create_degrees(n_inputs, n_hiddens, input_order, mode)
        self.Ms, self.Mmp = create_masks(degrees)
        self.Ws, self.bs, self.Wm, self.bm, self.Wp, self.bp = create_weights(n_inputs, n_hiddens, None)
        self.parms = self.Ws + self.bs + [self.Wm, self.bm, self.Wp, self.bp]
        self.input_order = degrees[0]

        # activation function
        self.f = self.act_fun

        # input matrix
        # self.input = tf.placeholder(dtype=dtype,shape=[None,n_inputs],name='x') if input is None else input
        self.input = input
        self.h = self.input

    def set_input(self, x):
        self.input = x
        self.h = self.input
        # feedforward propagation
        for l, (M, W, b) in enumerate(zip(self.Ms, self.Ws, self.bs)):
            self.h = self.f(tf.matmul(self.h, M * W) + b, name='h' + str(l + 1))

        # output means
    def m(self):
        return tf.add(tf.matmul(self.h, self.Mmp * self.Wm), self.bm, name='m')

        # output log precisions
    def logp(self):
        return tf.add(tf.matmul(self.h, self.Mmp * self.Wp), self.bp, name='logp')

        # random numbers driving made
    def u(self):
        return tf.exp(0.5 * self.logp()) * (self.input - self.m())

        # log likelihoods
    def L(self):
        return tf.multiply(-0.5, self.n_inputs * tf.math.log(2 * np.pi) + tf.reduce_sum(self.u() ** 2 - self.logp(), axis=1,keepdims=True),name='L')

        # train objective
    def trn_loss(self):
        return -tf.reduce_mean(self.L(), name='trn_loss')

    def eval(self, x, log=True):
        """
        Evaluate log probabilities for given inputs.
        :param x: data matrix where rows are inputs
        :param sess: tensorflow session where the graph is run
        :param log: whether to return probabilities in the log domain
        :return: list of log probabilities log p(x)
        """
        self.set_input(x)
        lprob = self.L()
        return lprob if log else np.exp(lprob)

    def eval_comps(self, x):
        """
        Evaluate the parameters of all gaussians at given input locations.
        :param x: rows are input locations
        :param sess: tensorflow session where the graph is run
        :return: means and log precisions
        """
        self.set_input(x)
        return self.m(),self.logp()

    def gen(self, n_samples=1, u=None):
        """
        Generate samples from made. Requires as many evaluations as number of inputs.
        :param sess: tensorflow session where the graph is run
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        x = np.zeros([n_samples, self.n_inputs])
        u = rng.randn(n_samples, self.n_inputs) if u is None else u

        for i in range(1, self.n_inputs + 1):
            m, logp = self.eval_comps(x)
            idx = np.argwhere(self.input_order == i)[0, 0]
            x[:, idx] = m[:, idx] + np.exp(np.minimum(-0.5 * logp[:, idx], 10.0)) * u[:, idx]

        return x
