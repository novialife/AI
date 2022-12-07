 

def efficient_backward(self, O, C_list):
        B = np.zeros((self.N, self.T))
        B[:, -1] = 1
        for t in range(self.T-2, -1, -1):
            B[:, t] = np.logaddexp.reduce(np.log(B[:, t+1]) + np.log(self.B[:, O[t+1]]) + np.log(self.A), axis=1)
            B[:, t] = B[:, t] * C_list[t]
        return B

def backward_testing(self, O, C_list):
        B = np.zeros((self.N, self.T))
        B[:, -1] = 1
        for t in range(self.T-2, -1, -1):
            B[:, t] = np.logaddexp.reduce(np.log(B[:, t+1]) + np.log(self.B[:, O[t+1]]) + np.log(self.A), axis=1)
            B[:, t] = B[:, t] * C_list[t]
        return B