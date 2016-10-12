

import numpy as np

class TimeEvolutionED(object):
    """docstring for TimeEvolutionED"""
    def __init__(self, model):
        self.model = model

    def evolve_to(self, initial_states, t):
        """
        |\psi_t> = sum_alpha <alpha|psi_0> e^{-i*Ea*t}*|alpha>
        Initial states must be a matrix of the form (HDim, N_states)
        returns: final states, where each column corresponds to a different initial state
        """
        phases_t = np.exp(-1.0j*self.model.eig_vals*t)
        projections = np.asmatrix( (self.model.eig_vecs.H.A*phases_t[:, np.newaxis]) )*initial_states # <alpha|psi_0> e^{-i*Ea*t}
        final_states = self.model.eig_vecs*projections # final states in the columns
        return final_states


    def infinite_time_average(self, op, initial_states):
        """
        \sum_\alpha O_\alpha |A_\alpha|**2
        returns: Infinite time average of operator op, the nth element corresponds to the nth initial state

        """
        
        op_diag_alpha = np.diag(self.model.eig_vecs.H*op*self.model.eig_vecs)
        projections = self.model.eig_vecs.H*initial_states
        projections_sq = np.asmatrix(np.abs(projections.T.A)**2)
        # projections_sq = (np.abs(np.asmatrix((self.eig_vecs.H.A*self.eig_vals[:, np.newaxis]))*initial_states)**2).T # each row corresponds to the projs of one state
        infinite_time_avg = projections_sq*op_diag_alpha[:, np.newaxis]
        return infinite_time_avg.A1
