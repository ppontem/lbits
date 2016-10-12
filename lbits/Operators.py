import numpy as np




class Operators(object):
    """docstring for Operators"""
    def __init__(self, op):
        super(Operators, self).__init__()
        self.op = op

    def expectation(self, states):
        return np.diag(states.H*self.op*states)