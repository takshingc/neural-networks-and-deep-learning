"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

class RNN:
    '''Recurrent Neural Network Model

    Args:
        size (int): number of hidden state neurons
    '''

    def __init__(self, size):
        self.size = size

    def train(self, Xs, seq_length = 25, learning_rate = 1e-1):

        data_size = Xs.shape[1]
        self.U = np.random.randn(self.size, data_size)*0.01 # input to hidden
        self.W = np.random.randn(self.size, self.size)*0.01 # hidden to hidden
        self.V = np.random.randn(data_size, self.size)*0.01 # hidden to output
        self.b_s = np.zeros((self.size, 1)) # hidden bias
        self.b_o = np.zeros((data_size, 1)) # output bias

        # TODO: cut Xs into mini batches
        for mini_batch in mini_batches:
            states, os = self.forward(Xs)
            dU, dW, dV, db_s, db_o = backprop(Xs, states, os, ys)

            # perform parameter update with Adagrad
            mU, mW, mV = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
            mb_s, mb_o = np.zeros_like(b_s), np.zeros_like(b_o) # memory variables for Adagrad
            for param, dparam, mem in zip([self.U, self.W, self.V, self.b_s, self.b_o],
                                            [dU, dW, dV, db_s, db_o],
                                            [mU, mW, mV, mb_s, mb_o]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-7) # adagrad update, the change will become smaller over time

    def forward(self, Xs):
        states, os = [], []
        state = 0
        for index, x in enumerate(Xs):
            states[index] = state = np.tanh(np.dot(self.U, x) + np.dot(self.W, state) + bh)
            net_o = np.dot(V, state) + by
            os[index] = o = np.exp(y) / np.sum(np.exp(net_o)) # probability distribution

        return states, os

    def backprop(Xs, states, os, ys):
        '''
        Args:
            ys (np.array): one-hot encoded target array
        '''
        dU, dW, dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        db_s, db_o = np.zeros_like(self.b_s), np.zeros_like(self.b_o)
        ds_next = np.zeros_like(self.size)
        # should leave the last layer
        for t in range(len(ys), -1, -1):
            dnet_o = np.copy(os[t])
            dnet_o -= ys[t] # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here

            # output layer
            dV += np.dot(dnet_o, states[t].T)
            db_o += dnet_o

            # hidden layer
            # add the ds_next to this time period ds, ds_next is an accumulated value of ds
            # can think of ds as ds and ds_next stack together, not add together
            ds = np.dot(V.T, dnet_o) + ds_next # backprop into h
            dnet = (1 - states[t] * states[t]) * ds # backprop through tanh nonlinearity
            db_s += dnet

            # input weights
            dU += np.dot(dnet, Xs[t].T)

            # recurrent weights
            dW += np.dot(dnet, states[t-1].T)

            # error back propagation
            ds_next = np.dot(W.T, dnet)

        for dparam in [dU, dW, dV, db_s, db_o]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
        return dU, dW, dV, db_s, db_o

    def sample(state, n):
        """
        sample a sequence of integers from the model
        state is memory state
        """
        x = np.zeros((data_size, 1))
        x[random.randint(0, data_size-1)] = 1
        ys = []
        for t in range(n):
            s = np.tanh(np.dot(self.U, x) + np.dot(self.W, h) + self.b_s)
            net_o = np.dot(self.V, h) + self.b_o
            o = np.exp(net_o) / np.sum(np.exp(net_o))
            ix = np.random.choice(range(data_size), p=o)
            x = np.zeros((vocab_size, 1))
            x[ix] = 1
            ys = np.concatenate((ys, x), axis=1)
        return ys

# data I/O
def get_data():
    data = open('E:\\develop\\neural-networks-and-deep-learning\\rnn\\input.txt', 'r').read() # should be simple plain text file
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1


n, p = 0, 0
mWxh, mW, mV = np.zeros_like(Wxh), np.zeros_like(W), np.zeros_like(V)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p+seq_length+1 >= len(data) or n == 0: 
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))

    # forward seq_length characters through the net and fetch gradient
    loss, dU, dW, dV, db_s, db_o, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss)) # print progress

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, W, V, bh, by],
                                    [dU, dW, dV, db_s, db_o],
                                    [mWxh, mW, mV, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

    p += seq_length # move data pointer
    n += 1 # iteration counter
