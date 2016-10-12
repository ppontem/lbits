import numpy as np

class ParticipationRatios(object):

    def __init__(self, basis):
        self.basis = basis

    def compute(self, state):
        amp_abs = np.abs(self.basis.transpose().dot(state.conj()))
        return 1.0/np.sum(amp_abs**4, axis=0)


