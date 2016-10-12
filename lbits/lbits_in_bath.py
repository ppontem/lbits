import scipy.linalg
import numpy as np
import scipy.sparse
import LevelStatistics
import Entanglement
import ParticipationRatios
import scipy.spatial.distance


class lbitsInBath(object):

    sigma_dict = {"x": scipy.sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]]),
                  "y": scipy.sparse.csr_matrix([[0.0, -1.0j], [0.0, 1.0j]]),
                  "z": scipy.sparse.csr_matrix([[1.0, 0.0], [0.0, -1.0]])}

    def __init__(self, N, M):
        """
        :param N: number of l-bits
        :param M: number of spins in the bath
        """
        self.N = N  # Number of l-bits
        self.M = M  # Effective number of spins in bath
        self.deltas = None
        self.Js = None
        self.ham = None
        self.ham_lbits = None
        self.ham_bath = None
        self.ham_int = None
        self.eig_vals = None
        self.eig_vecs = None
        self.level_statistics_r = None
        self.lbits_entanglement = None
        self.lbits_entanglement_average = None
        self.bath_energy_width = None
        self.bath_level_spacing = None
        self.bath_eig_vecs = None
        self.bath_eig_vals = None
        self.lbits_energy_width = None
        self.interaction_type = None
        self.bath_eig_vals = None
        self.lbits_bath_noint_eigvals = None
        self.lbits_bath_noint_eigvecs = None

    def set_deltas(self, deltas):
        """
        Logitudinal fields in the hamiltonian for the l-bits: H=\sum_{i=1}^N deltas[i-1]*\sigma^z_{i}
        :param deltas:
        :return:
        """
        self.deltas = deltas
        self.ham_lbits = self.lbits_hamiltonian(self.N, self.deltas)
        self.lbits_eig_vals = np.diag(self.ham_lbits.toarray())
        self.lbits_energy_width = lbitsInBath.energy_width(self.ham_lbits)

    def set_Js(self, Js):
        """
        :param Js: List of couplings in the interaction hamiltonian between the l-bits and the bath.
        """
        self.Js = Js

    @staticmethod
    def sample_GOE_random_matrix_M_norm(M, seed):
        """
        Generates GOE random matrix with extensive norm ~M in the number of degrees of freedom.
        :param M: Number of spins in the bath.
        :param seed: seed to reproduce GOE sample.
        :return: sample Hamiltonian.
        """
        np.random.seed(seed)
        size = 2**M
        diagonal_elements = np.random.normal(loc=0.0, scale=np.sqrt(2*M**2/float(size)), size=size)
        off_diagonal_elements = [np.random.normal(loc=0.0, scale=np.sqrt(M**2/float(size)), size=diag_size) for diag_size in range(size-1,0,-1)]
        ham = scipy.sparse.diags(off_diagonal_elements, range(1, size), format='csr')
        ham = ham + ham.transpose()
        ham.setdiag(diagonal_elements)
        return ham

    @staticmethod
    def sample_GOE_random_matrix_unit_norm(M, seed):
        """
        Generates GOE random matrix with order 1 norm in the number of degrees of freedom.
        :param M: Number of spins in the bath.
        :param seed: seed to reproduce GOE sample.
        :return: sample Hamiltonian.
        """
        np.random.seed(seed)
        size = 2**M
        diagonal_elements = np.random.normal(loc=0.0, scale=np.sqrt(2.0/float(size)), size=size)
        off_diagonal_elements = [np.random.normal(loc=0.0, scale=np.sqrt(1.0/float(size)), size=diag_size) for diag_size in range(size-1,0,-1)]
        ham = scipy.sparse.diags(off_diagonal_elements, range(1, size), format='csr')
        ham = ham + ham.transpose()
        ham.setdiag(diagonal_elements)
        return ham


    @staticmethod
    def lbits_hamiltonian(N, deltas):
        """
        l-bits Hamiltonian H=\sum_{i=1}^N deltas[i-1]*\sigma^z_{i}
        :param N: number of l-bits
        :param deltas: longitudinal fields for the l-bits
        :return: Hamiltonian
        """
        Ham = scipy.sparse.csr_matrix((2**N, 2**N))
        for i in range(1, N+1):
            Ham = Ham + deltas[i-1]*lbitsInBath.sigma("z", i, N)
        return Ham

    @staticmethod
    def sigma(typ, i, L):
        """
        Generates pauli matrix of type typ in site i in the Hilbert space of L spins.
        :param typ: pauli matrix type "x", "y", "z"
        :param i:  site for pauli matrix
        :param L: system size
        """
        ham = scipy.sparse.kron(lbitsInBath.sigma_dict[typ], scipy.sparse.identity(2**(L-i)))
        ham = scipy.sparse.kron(scipy.sparse.identity(2**(i-1)), ham)
        return ham

    def build_bath(self, seed, type="constant_width"):
        """
        Builds the system bath: the width of the bath can be extensive in the number of degrees of freedom or have unit norm.
        :param seed: seed for random matrix
        :param type: type of bath in terms of scaling of the width of eigenvalues
        :return:
        """

        self.bath_type = type
        if self.bath_type == "constant_width":
            self.ham_bath = self.sample_GOE_random_matrix_unit_norm(self.M, seed)
        elif  self.bath_type == "extensive_width":
            self.ham_bath = self.sample_GOE_random_matrix_M_norm(self.M, seed)



    def build_interaction_hamiltonian(self, seed, interaction_type="sx"):
        """
        Builds the hamiltonian of the system of lbits and bath.
        seed: seed to generate GOE matrix
        interaction_type: the interaction between the bath and l-bits can be of type "sx", in this case \sigma^x_{i} \gamma^x_{1} (gammas are pauli for bath).
        Moreover, it can be of type "goe" \sigma^x_{i}*H_GOE_prime, where H_GOE is an operator of norm 1.
        :return: None
        """

        self.interaction_type = interaction_type
        if self.interaction_type == "sx":
            h_bits_bath = scipy.sparse.csr_matrix((2**(self.N + self.M), 2**(self.N + self.M)))
            for i in range(1, self.N + 1):
                int_ham = scipy.sparse.kron(self.sigma("x", i, self.N), self.sigma("x", 1, self.M))
                h_bits_bath = h_bits_bath + self.Js[i - 1] * int_ham

        elif self.interaction_type == "goe":
            # np.random.rand(seed + 12345)
            h_goe_unit_norm = lbitsInBath.sample_GOE_random_matrix_unit_norm(self.M, seed + 12345)
            # print h_goe_unit_norm
            h_bits_bath = scipy.sparse.csr_matrix((2**(self.N + self.M), 2**(self.N + self.M)))
            for i in range(1, self.N + 1):
                int_ham = scipy.sparse.kron(self.sigma("x", i, self.N), h_goe_unit_norm)
                h_bits_bath = h_bits_bath + self.Js[i - 1] * int_ham

        elif self.interaction_type == "X_oneLBit_goe":
            # np.random.rand(seed + 12345)
            h_goe_unit_norm = lbitsInBath.sample_GOE_random_matrix_unit_norm(self.M, seed + 12345)
            # print h_goe_unit_norm
            h_bits_bath = scipy.sparse.csr_matrix((2 ** (self.N + self.M), 2 ** (self.N + self.M)))
            for i in range(1, 2):
                int_ham = scipy.sparse.kron(self.sigma("x", i, self.N), h_goe_unit_norm)
                h_bits_bath = h_bits_bath + self.Js[i - 1] * int_ham

        else:
            raise ValueError('interaction not understood')
        self.ham_int = h_bits_bath
        h_lbits_full = scipy.sparse.kron(self.ham_lbits, scipy.sparse.identity(2**self.M))

        h_bath_full = scipy.sparse.kron(scipy.sparse.identity(2**self.N), self.ham_bath)

        self.ham = h_lbits_full + h_bath_full + h_bits_bath


    def diagonalize(self, save_eig_vecs=True):
        self.eig_vals, eig_vecs = scipy.linalg.eigh(self.ham.todense())
        if save_eig_vecs:
            self.eig_vecs = eig_vecs

    def diagonalize_bath(self):
        # self.bath_eig_vals = np.sort(scipy.linalg.eigvalsh(self.ham_bath.todense()))
        self.bath_eig_vals, self.bath_eig_vecs = scipy.linalg.eigh(self.ham_bath.todense())
        self.bath_energy_width = 2.0*np.std(self.bath_eig_vals)
        lvl_spacing = np.sort(self.bath_eig_vals)[1:] - np.sort(self.bath_eig_vals)[:-1]
        self.bath_level_spacing = [np.percentile(lvl_spacing, 25), np.median(lvl_spacing), np.percentile(lvl_spacing, 75)]
        # print "level"



    def get_level_statistics(self):
        """
        :return: List of statistical quantity r for the eigenvalues of the system
        """
        # print self.eig_vals[:10]
        lvl_stats = LevelStatistics.LevelStatistics(self.eig_vals, t='h')
        self.level_statistics_r = lvl_stats.get_r()
        return self.level_statistics_r

    def lbit_entanglement_all_eigenstates(self):
        """
        :return: List of entanglement entropies (in base 2) of each l-bit with the rest of the system.
        """
        ent_average = np.zeros(self.N)
        ent_eigs = []
        for eig_vec in np.transpose(self.eig_vecs):
            ent_eig = Entanglement.EntanglementEntropy(self.N + self.M, eig_vec).single_bit_entanglement(range(1, self.N+1)).values()
            ent_average += ent_eig
            ent_eigs.append(ent_eig)
            # print ent_eig
            # ent_average += Entanglement.EntanglementEntropy(self.N + self.M, eig_vec).single_bit_entanglement(range(1, self.N+1)).values()
        ent_average /= 2**(self.N + self.M)
        self.lbits_entanglement = np.asarray(ent_eigs)
        self.lbits_entanglement_average = ent_average

    # def lbit_entanglement_all_eigenstates(self):
    #     """
    #     :return: List of entanglement entropies (in base 2) of each l-bit with the rest of the system.
    #     """
    #     ent_average = np.zeros(self.N)
    #     ent_eigs = []
    #     for eig_vec in np.transpose(self.eig_vecs):
    #         ent_eig = Entanglement.EntanglementEntropy(self.N + self.M, eig_vec).single_bit_entanglement(range(1, self.N+1)).values()
    #         ent_average += ent_eig
    #         ent_eigs.append(ent_eig)
    #         # print ent_eig
    #         # ent_average += Entanglement.EntanglementEntropy(self.N + self.M, eig_vec).single_bit_entanglement(range(1, self.N+1)).values()
    #     ent_average /= 2**(self.N + self.M)
    #     self.lbits_entanglement = np.asarray(ent_eigs)
    #     self.lbits_entanglement_average = ent_average

    def subsystem_entanglement_all_eigenstates(self, bit_labels):
        """
        :return: List of entanglement entropy (in base 2) for subsystem composed by the spins in bit_labels (from 1 to N+M)
        """
        ent_eigs = []
        for eig_vec in np.transpose(self.eig_vecs):
            ent_eig = Entanglement.EntanglementEntropy(self.N + self.M, eig_vec).subsystem_entanglement(bit_labels)
            ent_eigs.append(ent_eig)
            # print ent_eig
            # ent_average += Entanglement.EntanglementEntropy(self.N + self.M, eig_vec).single_bit_entanglement(range(1, self.N+1)).values()
        return ent_eigs

    @staticmethod
    def energy_width(ham):
        ham_ = np.asmatrix(ham.todense())
        # print ham_
        width = (ham_*ham_).trace()/ham_.shape[0] - (ham_.trace()/ham_.shape[0])**2
        width = width[0, 0]
        return 2*np.sqrt(width)

    def lbit_entanglement_infTemp_eigvecs(self, n_vecs):
        inds = sorted(range(len(self.eig_vals)), key=lambda i: np.abs(self.eig_vals[i]-self.eig_vals.mean()))[:n_vecs]
        ent_average = np.zeros(self.N)
        ent_eigs = []
        for eig_vec in (np.transpose(self.eig_vecs)[inds]):
            ent_eig = Entanglement.EntanglementEntropy(self.N + self.M, eig_vec).single_bit_entanglement(
                range(1, self.N + 1)).values()
            # ent_average += ent_eig
            ent_eigs.append(ent_eig)
            # print ent_eig
            # ent_average += Entanglement.EntanglementEntropy(self.N + self.M, eig_vec).single_bit_entanglement(range(1, self.N+1)).values()
        # ent_average /= 2 ** (self.N + self.M)
        # self.lbits_entanglement = np.asarray(ent_eigs)
        # self.lbits_entanglement_average = ent_average
        return np.asarray(ent_eigs)

    def entanglement_btwn_lbits_and_bath(self, n_vecs):
        inds = sorted(range(len(self.eig_vals)), key=lambda i: np.abs(self.eig_vals[i] - self.eig_vals.mean()))[:n_vecs]
        ent_eigs = []
        for eig_vec in (np.transpose(self.eig_vecs)[inds]):
            ent_eig = Entanglement.EntanglementEntropy(self.N + self.M, eig_vec).bipartite_entanglement(self.N, self.M)
            ent_eigs.append(ent_eig)
        return ent_eigs

    def lbits_reduced_density_matrix(self, state):
        ent = Entanglement.EntanglementEntropy(self.N + self.M, state)
        rho = ent.reduced_density_matrix(self.N, self.M)
        return rho

    def lbits_reduced_density_matrix_hamming_distance(self, n_vecs):
        states = np.asarray([[int(x) for x in ('{0:0' + str(self.M) + 'b}').format(j)] for j in range(2 ** self.N)])
        inds = sorted(range(len(self.eig_vals)), key=lambda i: np.abs(self.eig_vals[i] - self.eig_vals.mean()))[:n_vecs]
        hamming_distance_probs_collect = []
        diag_exp_collect = []
        for eig_vec in (np.transpose(self.eig_vecs)[inds]):
            rho = self.lbits_reduced_density_matrix(eig_vec)
            diag_exp = np.diag(rho)
            # print np.max(diag_exp)
            sorted_args = np.argsort(diag_exp)[::-1]
            average_ham = 0.0
            max_dist = self.N
            hamming_distance_probs = np.zeros(max_dist + 1)
            for j in range(2 ** self.N):
                hamming_dist = scipy.spatial.distance.hamming(states[sorted_args[0]], states[sorted_args[j]]) * self.N
                average_ham += hamming_dist * diag_exp[sorted_args[j]]
                hamming_distance_probs[int(hamming_dist)] += diag_exp[sorted_args[j]]
            hamming_distance_probs_collect.append(hamming_distance_probs.tolist())
            diag_exp_collect.append(np.sort(diag_exp)[::-1].tolist())
        return hamming_distance_probs_collect, diag_exp_collect


    def diagonalize_non_interacting_lbits_bath(self):
        h_lbits_full = scipy.sparse.kron(self.ham_lbits, scipy.sparse.identity(2 ** self.M))
        h_bath_full = scipy.sparse.kron(scipy.sparse.identity(2 ** self.N), self.ham_bath)
        self.lbits_bath_noint_eigvals, self.lbits_bath_noint_eigvecs = scipy.linalg.eigh(
            (h_lbits_full + h_bath_full).toarray())

    def perturbation_theory_amplitudes_random(self, n):
        V = self.ham_int
        V_me_eigs = np.transpose(self.lbits_bath_noint_eigvecs).conj().dot(V.dot(self.lbits_bath_noint_eigvecs)) # matrix elements of V between eigenstates
        # inds_i = sorted(range(len(self.lbits_bath_noint_eigvals)), key=lambda i: np.abs(self.lbits_bath_noint_eigvals[i] - self.lbits_bath_noint_eigvals.mean()))[:n]
        inds_i = np.random.choice(V.shape[0], n, replace=False)
        col_PT_amp =  []# collect the amplitude from first order perturbation theory V_{ki}/(E_k-E_i) each row corresponds to a given i
        for ind_i in inds_i:
            V_me_eigs_i = np.delete(V_me_eigs[:, ind_i], ind_i)
            energy_denom_i = np.delete(self.lbits_bath_noint_eigvals - self.lbits_bath_noint_eigvals[ind_i], ind_i)
            col_PT_amp.append(V_me_eigs_i/energy_denom_i)
        return np.asarray(col_PT_amp)

    def perturbation_theory_amplitudes_all(self):
        V = self.ham_int
        V_me_eigs = np.transpose(self.lbits_bath_noint_eigvecs).conj().dot(V.dot(self.lbits_bath_noint_eigvecs)) # matrix elements of V between eigenstates
        # inds_i = sorted(range(len(self.lbits_bath_noint_eigvals)), key=lambda i: np.abs(self.lbits_bath_noint_eigvals[i] - self.lbits_bath_noint_eigvals.mean()))[:n]
        # inds_i = np.random.choice(V.shape[0], n, replace=False)
        col_PT_amp =  []# collect the amplitude from first order perturbation theory V_{ki}/(E_k-E_i) each row corresponds to a given i
        for i in range(V.shape[0]):
            V_me_eigs_i = np.delete(V_me_eigs[:, i], i)
            energy_denom_i = np.delete(self.lbits_bath_noint_eigvals - self.lbits_bath_noint_eigvals[i], i)
            col_PT_amp.append(V_me_eigs_i/energy_denom_i)
        return np.asarray(col_PT_amp)

    def perturbation_theory_best_energy_denominator(self):
        V = self.ham_int
        V_me_eigs = np.transpose(self.lbits_bath_noint_eigvecs).conj().dot(V.dot(self.lbits_bath_noint_eigvecs)) # matrix elements of V between eigenstates
        col_PT_amp =  []# collect the amplitude from first order perturbation theory V_{ki}/(E_k-E_i) each row corresponds to a given i
        for i in range(V.shape[0]):
            V_me_eigs_i = np.delete(V_me_eigs[:, i], i)
            energy_denom_i = np.delete(self.lbits_bath_noint_eigvals - self.lbits_bath_noint_eigvals[i], i)
            ind_smallest_denom_list = sorted(range(len(energy_denom_i)), key=lambda j: np.abs(energy_denom_i[j]))
            for ind_small in ind_smallest_denom_list:
                if np.abs(V_me_eigs_i[ind_small]) > 1.0*10**(-8):
                    ind_smallest_denom = ind_small
                    break

            # if np.abs(energy_denom_i[ind_smallest_denom]) < 10**(-8):
            #     print "energy_denom less than 10**-8"
            col_PT_amp.append(V_me_eigs_i[ind_smallest_denom]/energy_denom_i[ind_smallest_denom])
        # if np.any(np.logical_not(np.isfinite(col_PT_amp))):
        #     print "array not finite"
        return np.asarray(col_PT_amp)


    def participation_ratios_non_interacting_basis(self, normalized=False):
        if self.lbits_bath_noint_eigvals is not None:
            self.diagonalize_non_interacting_lbits_bath()
        pr = ParticipationRatios.ParticipationRatios(self.lbits_bath_noint_eigvecs)
        if normalized:
            return pr.compute(self.eig_vecs)/float(len(self.eig_vecs))
        else:
            return pr.compute(self.eig_vecs)