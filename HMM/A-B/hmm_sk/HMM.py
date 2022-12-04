import sys
import math
import numpy as np
from constants import *


import sys
import math

epsilon = sys.float_info.epsilon

def make_matrix(matrix, shape):
    m = []
    for i in range(shape[0]):
        m.append(matrix[i*shape[1]:(i+1)*shape[1]])
    return m

# Matrix multiplication for nxm and mxp matrices
def multiply_matrices(A, B):

    return [[sum(a * b for a, b in zip(a_row, b_col)) for b_col in zip(*B)] for a_row in A]

def dot_product(A, B):

    return [[a * b for a, b in zip(A[0], B)]]

def transpose(A):

    return list(map(list, zip(*A)))

class HMM:
    def __init__(self) -> None:
        self.A = None
        self.B = None
        self.pi = None
        self.O = None
        self.N = None
        self.T = None        
    
    def __str__(self) -> str:
        return f"A: {self.A} \nB: {self.B} \npi: {self.pi}, \nO: {self.O}, \nN: {self.N}, \nT: {self.T}, \nM: {self.M}, \n"

    def forward(self):
        alpha = []
        p = []        

        for t in range(self.T):
            p_t = 0
            alpha_t = []
            for n in range(self.N):
                if t == 0:
                    _ = self.pi[0][n] * self.B[n][self.O[t]]
                    alpha_t.append(_)
                    p_t += _
                else:
                    alpha_t_t = sum([alpha[t-1][_] * self.A[_][n] * self.B[n][self.O[t]] for _ in range(self.N)])
                    alpha_t.append(alpha_t_t)
                    p_t += alpha_t_t

            alpha_t = [alpha_t[n] * (1/p_t) for n in range(self.N)]
            p.append(1/(p_t + epsilon))
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
                    beta_t_t = sum([beta[t-1][_] * self.A[n][_] * self.B[_][O[t-1]] for _ in range(self.N)])
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
                gamma_ij_t_n = [alpha[t][n] * self.A[n][_] * self.B[_][O[t+1]] * beta[t+1][_] for _ in range(self.N)]
                gamma_t_n = sum(gamma_ij_t_n)
                gamma_ij_t.append(gamma_ij_t_n)
                gamma_t.append(gamma_t_n)
            gamma.append(gamma_t)
            gamma_ij.append(gamma_ij_t)
        
        alpha_t = alpha[t+1]
        gamma.append([alpha_t[n] for n in range(self.N)])
        return gamma, gamma_ij

    def reestimate(self, gamma, gamma_ij, O):
        # Reestimate A
        for i in range(self.N):
            for j in range(self.N):
                numer = sum([gamma_ij[t][i][j] for t in range(self.T-1)])
                denom = sum([gamma[t][i] for t in range(self.T-1)])
                self.A[i][j] = numer / (denom + epsilon)

        # Reestimate B
        for i in range(self.N):
            for j in range(self.M):
                numer = sum([gamma[t][i] for t in range(self.T) if O[t] == j])
                denom = sum([gamma[t][i] for t in range(self.T)])
                self.B[i][j] = numer / (denom + epsilon)

        # Reestimate pi
        self.pi = [[gamma[0][i] for i in range(self.N)]]
    
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
                break
            log_p = self.prob_log(p)
        
    def guess(self, fish):
        obs = transpose(self.B)
        alpha = dot_product(self.pi, obs[fish[0]])

        for e in fish[1:]:
            alpha = multiply_matrices(alpha, self.A)
            alpha = dot_product(alpha, obs[e])

        return sum(alpha[0])


    def init_parameters(self, N, M):
        self.A = np.random.dirichlet(np.ones((N)), size=(N))
        self.B = np.random.dirichlet(np.ones((M)), size=(N))
        self.pi = [[1/N for _ in range(N)]]
        self.N = len(self.A)
        self.M = M
