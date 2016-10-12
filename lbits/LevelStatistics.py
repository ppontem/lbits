import numpy as np

class LevelStatistics:
    def __init__(self, vals, t='f'):
        self.vals = np.copy(vals)
        self.type = t
    def get_r(self):
        if self.type == 'f':
            phases = np.real(1.0j*np.log(self.vals))
            phases.sort()
            gaps = phases[1:] - phases[:-1]
            r = []
            for i in range(1, len(gaps)):
                r_i = min(gaps[i-1], gaps[i])/max(gaps[i-1], gaps[i])
                r.append(r_i)
            return np.array(r)
        elif self.type == 'h':
            vals_SORTED = np.sort(self.vals)
            gaps = vals_SORTED[1:] - vals_SORTED[:-1]
            r = []
            for i in range(1, len(gaps)):
                r_i = min(gaps[i-1], gaps[i])/max(gaps[i-1], gaps[i])
                r.append(r_i)
            return np.array(r)
        else:
            return "Type of the problem is neither floquet or hamiltonian"

    @staticmethod
    def goe_prediction(r):
        return 2*27.0/8.0*(r + r**2)/(1 + r + r**2)**(2.5)

