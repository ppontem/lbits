import numpy as np
import scipy.sparse

sigma_dict = {"x": [[0.0, 1.0],[1.0, 0.0]], "y":[[0.0, -1.0j],[0.0, 1.0j]], "z":[[1.0, 0.0],[0.0, -1.0]]}

# print np.kron(sigma_dict["x"], sigma_dict["z"])
print np.kron(sigma_dict["z"], np.kron( np.eye(2), sigma_dict["z"]))
print np.kron(np.eye(2), sigma_dict["z"])

a = scipy.sparse.csr_matrix((2,2))
a = a + scipy.sparse.csr_matrix(sigma_dict["x"])
print a
b = np.array([1, 2, 3])
b[b<2] = -4
print b

b = np.array([1,2,3,4])
print b.reshape((2,2))

import Entanglement
state = np.zeros(2**4)
state[0] = 1.0
system = Entanglement.EntanglementEntropy(4, state)
print system.single_bit_entanglement([1,2,3,4])

