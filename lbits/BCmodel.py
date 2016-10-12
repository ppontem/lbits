import scipy.linalg
import numpy as np
import scipy.sparse
import LevelStatistics
import Entanglement
import ParticipationRatios
import scipy.spatial.distance

from lbits_in_bath import lbitsInBath


class BCmodel(lbitsInBath):

    def __init__(self, N, M, delta_list, J_list, bath_seed, interactions_seed):

        """
        :param N: number of l-bits
        :param M: number of "degrees of freedom" in the bath -- the dimension of the hilbert space of the bath is 2**M.
        :param delta_list: list of z-fields applied to the l-bits
        :param J_list: list of couplings between each l-bit and the bath
        :param bath_seed: seed for the bath random matrix hamiltonian
        :param interactions_seed: seed for the random matrices involved in the interactions
        """

        lbitsInBath.__init__(self, N, M)
        self.set_deltas(delta_list)  # build the l-bits hamiltonian
        self.build_bath(bath_seed, type="constant_width")  # build the bath
        self.set_Js(J_list)  # couplings of the l-bits to the bath
        self.build_interaction_hamiltonian(interactions_seed)


    @staticmethod
    def sample_goe_random_matrix_unit_norm_no_seed(M):
        """
        :param M: Number of spins in the bath
        :return: sample Hamiltonian from GOE with norm that does not scale with system size
        """
        # np.random.seed(seed)
        size = 2**M
        diagonal_elements = np.random.normal(loc=0.0, scale=np.sqrt(2.0/float(size)), size=size)
        off_diagonal_elements = [np.random.normal(loc=0.0, scale=np.sqrt(1.0/float(size)), size=diag_size) for diag_size in range(size-1,0,-1)]
        ham = scipy.sparse.diags(off_diagonal_elements, range(1, size), format='csr')
        ham = ham + ham.transpose()
        ham.setdiag(diagonal_elements)
        return ham

    @staticmethod
    def sample_goe_random_matrix_unit_norm_no_seed_notsparse(M):
        """
        :param M: Number of spins in the bath
        :return: sample Hamiltonian from GOE with norm that does not scale with system size
        """
        # np.random.seed(seed)
        size = 2**M
        ham = 1/np.sqrt(2.0)*np.random.randn(size, size)
        ham = ham + np.transpose(ham)
        ham[range(size), range(size)] = np.sqrt(2.0)*np.random.randn(size)
        return ham/np.sqrt(2**M)

    def build_interaction_hamiltonian(self, seed):
        """
        Builds the interaction part of the Hamiltonian between the l-bits and the bath; The interaction is
        of the following form: \sum_{i=1}^N J_i (\tau^z_i B_i + \tau^x_i C_i).
        :param seed: initializes the random generator which builds the different random matrices
        """

        np.random.seed(seed + 1234567)
        h_bits_bath = scipy.sparse.csr_matrix((2 ** (self.N + self.M), 2 ** (self.N + self.M)))
        for i in range(1, self.N + 1):
            Bi = self.sample_goe_random_matrix_unit_norm_no_seed(self.M)
            Ci = self.sample_goe_random_matrix_unit_norm_no_seed(self.M)
            int_ham_x = scipy.sparse.kron(self.sigma("x", i, self.N), Ci)
            int_ham_z = scipy.sparse.kron(self.sigma("z", i, self.N), Bi)
            h_bits_bath = h_bits_bath + self.Js[i - 1] * (int_ham_x + int_ham_z)
            if i == 1:
                self.C = [Bi]
                self.B = [Ci]
            else:
                self.C.append(Bi)
                self.B.append(Ci)


        self.ham_int = h_bits_bath
        h_lbits_full = scipy.sparse.kron(self.ham_lbits, scipy.sparse.identity(2 ** self.M))
        h_bath_full = scipy.sparse.kron(scipy.sparse.identity(2 ** self.N), self.ham_bath)
        self.ham = h_lbits_full + h_bath_full + h_bits_bath


    def build_interaction_hamiltonian_not_sparse(self, seed):
        """
        Builds the interaction part of the Hamiltonian between the l-bits and the bath; The interaction is
        of the following form: \sum_{i=1}^N J_i (\tau^z_i B_i + \tau^x_i C_i).
        :param seed: initializes the random generator which builds the different random matrices
        """

        np.random.seed(seed + 1234567)
        h_bits_bath = np.zeros((2 ** (self.N + self.M), 2 ** (self.N + self.M)))
        for i in range(1, self.N + 1):
            Bi = self.sample_goe_random_matrix_unit_norm_no_seed_notsparse(self.M)
            Ci = self.sample_goe_random_matrix_unit_norm_no_seed_notsparse(self.M)
            int_ham_x = np.kron(self.sigma("x", i, self.N).toarray(), Ci)
            int_ham_z = np.kron(self.sigma("z", i, self.N).toarray(), Bi)
            h_bits_bath = h_bits_bath + self.Js[i - 1] * (int_ham_x + int_ham_z)
            if i == 1:
                self.C = Bi
                self.B = Ci

        self.ham_int = h_bits_bath
        h_lbits_full = np.kron(self.ham_lbits.toarray(), np.identity(2 ** self.M))
        h_bath_full = np.kron(np.identity(2 ** self.N), self.ham_bath.toarray())
        self.ham = h_lbits_full + h_bath_full + h_bits_bath


# class BCmodelXY(BCmodel):

#     def __init__(self, N, M, delta_list, J_list, bath_seed, interactions_seed):

#         """
#         :param N: number of l-bits
#         :param M: number of "degrees of freedom" in the bath -- the dimension of the hilbert space of the bath is 2**M.
#         :param delta_list: list of z-fields applied to the l-bits
#         :param J_list: list of couplings between each l-bit and the bath
#         :param bath_seed: seed for the bath random matrix hamiltonian
#         :param interactions_seed: seed for the random matrices involved in the interactions
#         """

#         lbitsInBath.__init__(self, N, M)
#         self.set_deltas(delta_list)  # build the l-bits hamiltonian
#         self.build_bath(bath_seed, type="constant_width")  # build the bath
#         self.set_Js(J_list)  # couplings of the l-bits to the bath
#         self.build_interaction_hamiltonian_xy(interactions_seed)


#     def build_interaction_hamiltonian_xy(self, seed):
#         """
#         Builds the interaction part of the Hamiltonian between the l-bits and the bath; The interaction is
#         of the following form: \sum_{i=1}^N J_i (\tau^y_i B_i + \tau^x_i C_i).
#         :param seed: initializes the random generator which builds the different random matrices
#         """

#         np.random.seed(seed)
#         h_bits_bath = scipy.sparse.csr_matrix((2 ** (self.N + self.M), 2 ** (self.N + self.M)))
#         for i in range(1, self.N + 1):
#             Bi = self.sample_goe_random_matrix_unit_norm_no_seed(self.M)
#             Ci = self.sample_goe_random_matrix_unit_norm_no_seed(self.M)
#             int_ham_x = scipy.sparse.kron(self.sigma("x", i, self.N), Ci)
#             int_ham_y = scipy.sparse.kron(self.sigma("y", i, self.N), Bi)
#             h_bits_bath = h_bits_bath + self.Js[i - 1] * (int_ham_x + int_ham_y)

#         self.ham_int = h_bits_bath
#         h_lbits_full = scipy.sparse.kron(self.ham_lbits, scipy.sparse.identity(2 ** self.M))
#         h_bath_full = scipy.sparse.kron(scipy.sparse.identity(2 ** self.N), self.ham_bath)
#         self.ham = h_lbits_full + h_bath_full + h_bits_bath