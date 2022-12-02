import sys
import math
import numpy as np
from constants import *


class HMM:
    def __init__(self) -> None:
        self.A = None
        self.B = None
        self.pi = None
        self.O = None
        self.viterbi_res = None
        self.alpha = None
        self.T = None
    
    def efficient_forward(self):
        # Must calculate alpha using log exp to avoid underflow
        alpha = np.zeros((self.N, self.T))
        alpha[:, 0] = self.pi * self.B[:, self.O[0]]        
        C0 = np.sum(alpha[:, 0])
        alpha[:, 0] = alpha[:, 0] / C0
        C_list = [C0]
        for t in range(1, self.T):            
            alpha[:, t] = np.logaddexp.reduce((alpha[:, t-1]) + self.A, axis=1) + self.B[:, self.O[t]]
            C = np.sum(alpha[:, t])
            alpha[:, t] = alpha[:, t] / C
            C_list.append(1/C)
        return alpha, C_list
        
    def efficient_backward(self, O, C_list):
        B = np.zeros((self.N, self.T))
        B[:, -1] = 1
        for t in range(self.T-2, -1, -1):
            B[:, t] = np.logaddexp.reduce(B[:, t+1] + self.B[:, O[t+1]] + self.A, axis=1)
            B[:, t] = B[:, t] * C_list[t]
        return B
    
    def efficient_gamma(self, O, alpha, beta):
        # Compute gamma and gamma_ij for each state with numpy
        gamma_i = []
        gamma_ij = []

        for t in range(self.T-1):
            for i in range(self.N):
                for j in range(self.N):
                    gamma_ij_t = alpha[i][t] * self.A[i][j] * self.B[j][O[t+1]] * beta[j][t+1]
                    gamma_ij.append(gamma_ij_t)
                    gamma_i.append(np.sum(gamma_ij_t))

        for i in range(self.N):
            gamma_i.append(alpha[i, self.T-1])
        
        return gamma_i, gamma_ij

    def reestimate(self, gamma, gamma_ij, O):
        # Reestimate A
        for i in range(self.N):
            for j in range(self.N):
                self.A[i][j] = np.sum(gamma_ij[i*self.N+j::self.N]) / np.sum(gamma[i::self.N])

        # Reestimate B
        for i in range(self.N):
            for j in range(len(set(self.O))):
                self.B[i][j] = np.sum(gamma[i::self.N][O==j]) / np.sum(gamma[i::self.N])

        # Reestimate pi
        for i in range(self.N):
            self.pi[i] = gamma[i] / np.sum(gamma)  

    def prob_log(self, p):
        # Compute log probability
        return -np.sum(np.log(p))

    def baum_welch(self, max_iter):
        log_p = -np.inf
        for _ in range(max_iter):
            alpha, p = self.efficient_forward()
            p = p[::-1]
            obs = self.O[::-1]
            beta = self.efficient_backward(obs, p)
            beta = beta[::-1]

            gamma, gamma_ij = self.efficient_gamma(self.O, alpha, beta)
            self.reestimate(gamma, gamma_ij, self.O)

            if self.prob_log(p) < log_p:
                log_p = self.prob_log(p)
                return True
            log_p = self.prob_log(p)
        return False

    def init_parameters(self, N, M):
        self.A = np.random.uniform(0, 1, (N, N))
        self.A = self.A / np.sum(self.A, axis=1, keepdims=True)
        self.B = np.random.uniform(0, 1, (N, M))
        self.B = self.B / np.sum(self.B, axis=1, keepdims=True)

        # self.A = np.random.dirichlet(np.ones((N)), size=(N))
        # self.B = np.random.dirichlet(np.ones((M)), size=(N))
        self.pi = [1/N for _ in range(N)]
        self.N = len(self.A)