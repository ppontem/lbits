import numpy as np
import scipy.linalg
import collections


class EntanglementEntropy(object):
    """
    entanglement entropy class
    """
    def __init__(self, system_size, state):
        self.system_size = system_size
        self.state = state


    def single_bit_entanglement(self, bit_labels):
        """
        :param bit_labels: labels of the bits for which we compute the entanglement with the rest of the system (from 1
        to self.system_size)
        :return: OrderedDict with each bit as key and entanglement entropies as values.
        """
        ent_entropy_all_bits = collections.OrderedDict()
        for bit_label in bit_labels:
            if (bit_label == 1):
                shape_new = (2, 2**(self.system_size - 1))
                reshaped_state = np.reshape(self.state, shape_new)

            elif (bit_label == self.system_size):
                shape_new = (2**(self.system_size - 1), 2)
                reshaped_state = np.reshape(self.state, shape_new)

            else:
                shape_new = tuple([2**(bit_label-1)] + [2] + [2**(self.system_size - bit_label)])
                # print "shape_new", shape_new
                reshaped_state = np.reshape(self.state, shape_new)
                reshaped_state = np.transpose(reshaped_state, (1, 0, 2))
                reshaped_state = np.reshape(reshaped_state, (2, 2 ** (self.system_size - 1)))

            sing_vals = scipy.linalg.svdvals(reshaped_state)
            # print sing_vals
            probs = (sing_vals.real)**2
            # print "sum probs. :", np.sum(probs)
            # sing_vals[sing_vals < 0.0] = 1e-15
            ent_entropy_all_bits[bit_label] = 0.0
            # print
            for p in probs:
                if p > 0.0:
                    ent_entropy_all_bits[bit_label] += -p*np.log2(p)
        return ent_entropy_all_bits

    def subsystem_entanglement(self, bit_labels):
        """
        :param bit_labels: labels of the bits for which we compute the entanglement with the rest of the system (from 1
        to self.system_size)
        :return: OrderedDict with each bit as key and entanglement entropies as values.
        """
        bit_labels_inds = [i-1 for i in bit_labels]
        other_labels_inds = list(set(range(self.system_size)) - set(bit_labels_inds))
        state_reshape = np.reshape(self.state, tuple([2]*self.system_size))
        state_reshape = np.transpose(state_reshape, bit_labels_inds + other_labels_inds)
        state_reshape = np.reshape(state_reshape, (2**len(bit_labels_inds), 2**len(other_labels_inds)))
        sing_vals = scipy.linalg.svdvals(state_reshape)
        probs = (sing_vals.real)**2
        ent_entropy = 0.0
        # print
        for p in probs:
            if p > 0.0:
                ent_entropy += -p * np.log2(p)
        return ent_entropy

    def subsystem_reduced_density_matrix(self, bit_labels):
        """
        :param bit_labels: labels of the bits for which we compute the reduced density matrix  (from 1
        to self.system_size)
        :return: Reduced density matrix
        """
        bit_labels_inds = [i-1 for i in bit_labels]
        other_labels_inds = list(set(range(self.system_size)) - set(bit_labels_inds))
        state_reshape = np.reshape(self.state, tuple([2]*self.system_size))
        state_reshape = np.transpose(state_reshape, bit_labels_inds + other_labels_inds)
        state_reshape = np.reshape(state_reshape, (2**len(bit_labels_inds), 2**len(other_labels_inds)))
        state_reshape = np.asmatrix(state_reshape)
        rho = state_reshape*state_reshape.H
        return rho


    def bipartite_entanglement(self, left_bits, right_bits):
        """
        :param bit_labels: labels of the bits for which we compute the entanglement with the rest of the system (from 1
        to self.system_size)
        :return: OrderedDict with each bit as key and entanglement entropies as values.
        """

        ent_entropy_all_bits = collections.OrderedDict()
        reshaped_state = np.reshape(self.state, (2**left_bits, 2**right_bits)) # with kron last spins change first in the basis ordering, reshape fills along each row
        sing_vals = scipy.linalg.svdvals(reshaped_state)
        # print sing_vals
        probs = sing_vals.real**2
        ent = 0.0
        for p in probs:
            if p > 0.0:
                ent += -p * np.log2(p)
        return ent


    def reduced_density_matrix(self, left_bits, right_bits, which='left'):
        """
        :param bit_labels: labels of the bits for which we compute the entanglement with the rest of the system (from 1
        to self.system_size)
        :return: OrderedDict with each bit as key and entanglement entropies as values.
        """

        reshaped_state = np.reshape(self.state, (2**left_bits,
                                                 2**right_bits))  # with kron last spins change first in the basis ordering, reshape fills along each row
        reshaped_state = np.asmatrix(reshaped_state)
        if which == 'left':
            rho = reshaped_state*reshaped_state.H
        return rho


