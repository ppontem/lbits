import BCmodel
import numpy as np
import sys


class LBits_All_Z(object):
    """
    All l-bits up and eigenstates of the model.
    Requires bath_eig_vecs to be defined.
    """
    def __init__(self, model):
        self.model = model

    def prepare_states(self):
        try:
            if self.model.bath_eig_vecs != None:
                pass
        except:
            print "model.bath_eig_vecs not defined..." 
            sys.exit(1)
        try:
            if self.model.eig_vecs != None:
                pass
        except:
            print "model.eig_vecs not defined..." 
            sys.exit(1)
        initial_states = np.zeros((self.model.eig_vecs.shape[0], self.model.bath_eig_vecs.shape[1]))
        z1 = np.array([1.0, 0.0])
        if self.model.N > 1:
            for i in range(2, self.model.N + 1):
                z1 = np.kron(z1, np.array([1.0, 0.0]))
#         lbits_z = z1

        for i_, bath_eig_state in enumerate(np.transpose(self.model.bath_eig_vecs)):
#             print lbits_z.shape, bath_eig_state.shape, initial_states.shape
#             print bath_eig_state
            initial_states[:, i_] = np.kron(z1, bath_eig_state)
#             print initial_states[:, i_]
        return np.asmatrix(initial_states)


class LBits_2Correlated(object):
    """
    All l-bits up and eigenstates of the model.
    Requires bath_eig_vecs to be defined.
    """
    def __init__(self, model):
        self.model = model

    def prepare_states(self, typ="corr"):
        try:
            if self.model.bath_eig_vecs != None:
                pass
        except:
            print "model.bath_eig_vecs not defined..." 
            sys.exit(1)
        try:
            if self.model.eig_vecs != None:
                pass
        except:
            print "model.eig_vecs not defined..." 
            sys.exit(1)
        initial_states = np.zeros((self.model.eig_vecs.shape[0], self.model.bath_eig_vecs.shape[1]))
        if self.model.N > 1:
            if typ == "corr":
                z = (np.kron(np.array([1.0, 0.0]), np.array([1.0, 0.0])) + np.kron(np.array([0.0, 1.0]), np.array([0.0, 1.0])))/np.sqrt(2.0)
            elif typ == "acorr":
                z = (np.kron(np.array([1.0, 0.0]), np.array([0.0, 1.0])) + np.kron(np.array([0.0, 1.0]), np.array([1.0, 0.0])))/np.sqrt(2.0)
            else:
                print "typ not understood"
                sys.exit(1)
        else:
            print "These states need more than 1 l-bit"
            sys.exit(1)
        
        if self.model.N > 2:
            for i in range(3, self.model.N + 1):
                z = np.kron(z, np.array([1.0, 0.0]))

        z1=z
#         print z1
#         print np.sum(np.abs(z1)**2)
        for i_, bath_eig_state in enumerate(np.transpose(self.model.bath_eig_vecs)):
#             print lbits_z.shape, bath_eig_state.shape, initial_states.shape
#             print bath_eig_state
            initial_states[:, i_] = np.kron(z1, bath_eig_state)
#             print initial_states[:, i_]
        return np.asmatrix(initial_states)








