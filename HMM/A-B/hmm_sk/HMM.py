import sys
import math
import numpy as np
from constants import *


def make_matrix(matrix, shape):
    m = []
    for i in range(shape[0]):
        m.append(matrix[i*shape[1]:(i+1)*shape[1]])
    return m

# Matrix multiplication for nxm and mxp matrices
def multiply_matrices(A, B):
    n = len(A)
    m = len(B)
    p = len(B[0])
    C = [[0 for i in range(p)] for j in range(n)]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C

class HMM:
    def __init__(self) -> None:
        self.A = None
        self.B = None
        self.pi = None
        self.O = None
        self.viterbi_res = None
        self.alpha = None
        self.T = None
        
    def forward(self):
        alpha = []
        p = []

        for t in range(self.T):
            p_t = 0
            alpha_t = []
            pi_t = self.pi[0]
            for n in range(self.N):
                if t == 0:
                    _ = pi_t[n] * self.B[n][self.O[t]]
                    alpha_t.append(self.B[n][self.O[t]] * pi_t[n])
                    p_t += _
                else:
                    alpha_t_t = 0
                    for _ in range(self.N):
                        alpha_t_t += alpha[t-1][_] * self.A[_][n] * self.B[n][self.O[t]]
                    alpha_t.append(alpha_t_t)
                    p_t += alpha_t_t

            for n in range(self.N):
                alpha_t[n] = alpha_t[n] * (1/p_t)     

            p.append(1/p_t)
            alpha.append(alpha_t)

        return alpha, p

    def backward(self, O, p):
        beta = []
        for t in range(self.T):
            beta_t = []
            for n in range(self.N):
                if t == 0:
                    beta_t.append(p[t])
                else:
                    beta_t_t = 0
                    for _ in range(self.N):
                        beta_t_t += beta[t-1][_] * self.A[n][_] * self.B[_][O[t-1]]
                    beta_t.append(beta_t_t)
            if t == 0:
                beta.append(beta_t)
            else:
                beta.append([p[t] * beta_t[x] for x in range(self.N)])
            
        return beta
    
    def gamma(self, O, alpha, beta):
        # Compute gamma and gamma_ij for each state
        gamma = []
        gamma_ij = []
        for t in range(self.T-1):
            gamma_t = []
            gamma_ij_t = []
            for n in range(self.N):
                gamma_t_n = 0
                gamma_ij_t_n = []
                for _ in range(self.N):
                    gamma_ij_t_n_ = alpha[t][n] * self.A[n][_] * self.B[_][O[t+1]] * beta[t+1][_]
                    gamma_ij_t_n.append(gamma_ij_t_n_)
                    gamma_t_n += gamma_ij_t_n_
                gamma_ij_t.append(gamma_ij_t_n)
                gamma_t.append(gamma_t_n)
            gamma.append(gamma_t)
            gamma_ij.append(gamma_ij_t)
        gamma_t = []
        alpha_t = alpha[t+1]
        for n in range(self.N):
            gamma_t.append(alpha_t[n])
        gamma.append(gamma_t)
        return gamma, gamma_ij

    def reestimate(self, gamma, gamma_ij, O):
        # Reestimate A
        for i in range(self.N):
            for j in range(self.N):
                numer = sum([gamma_ij[t][i][j] for t in range(self.T-1)])
                denom = sum([gamma[t][i] for t in range(self.T-1)])
                self.A[i][j] = numer / denom

        # Reestimate B
        for i in range(self.N):
            for j in range(len(set(self.O))):
                numer = sum([gamma[t][i] for t in range(self.T) if O[t] == j])
                denom = sum([gamma[t][i] for t in range(self.T)])
                self.B[i][j] = numer / denom

        # Reestimate pi
        for i in range(self.N):
            self.pi[0][i] = gamma[0][i]
        
    
    def prob_log(self, p):
        log_p = 0
        for t in range(self.T):
            log_p -= math.log(p[t])
        return log_p


    def baum_welch(self, max_iter):
        log_p = -math.inf
        for _ in range(max_iter):
            alpha, p = self.forward()
            p = p[::-1]
            obs = self.O[::-1]
            beta = self.backward(obs, p)
            beta = beta[::-1]

            gamma, gamma_ij = self.gamma(self.O, alpha, beta)
            self.reestimate(gamma, gamma_ij, self.O)

            if self.prob_log(p) < log_p:
                log_p = self.prob_log(p)
                return True
            log_p = self.prob_log(p)
        return False

    def viterbi(self, delta, delta_idx, O):
        if len(O) == 0:
            seq = []

            last = delta.index(max(delta))
            seq.append(last)

            for i in range(len(delta_idx) - 1, 0, -1):
                seq.insert(0, delta_idx[i][last])
                last = delta_idx[i][last]

            self.viterbi_res = ' '.join([str(x) for x in seq])
            return 

        p = []
        for i in range(len(self.A)):
            _ = []
            for j in range(len(self.A)):
                _.append(self.A[j][i] * self.B[i][O[0]] * delta[j])
            p.append(_)

        max_p = []
        for i in p:
            max_p.append(max(i))
        _ = []
        for i, j in enumerate(p):
            _.append(j.index(max_p[i]))
        delta_idx.append(_)
        # Make delta_idx a NxN matrix where N is the number of states
        self.viterbi(max_p, delta_idx, O[1:])


    def init_parameters(self, N, M):
        self.A = np.random.dirichlet(np.ones((N)), size=(N))
        self.B = np.random.dirichlet(np.ones((M)), size=(N))
        self.pi = np.random.dirichlet(np.ones((N)), size=1)
        self.N = len(self.A)