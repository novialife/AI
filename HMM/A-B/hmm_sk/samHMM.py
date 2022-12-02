import sys
import math
import numpy as np
from constants import *


def make_matrix(matrix, shape):
    m = []
    for i in range(shape[0]):
        m.append(matrix[i*shape[1]:(i+1)*shape[1]])
    return m

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

    # Initialize the HMM
    def __init__(self) -> None:
        self.A = None
        self.B = None
        self.pi = None
        self.O = None
        self.viterbi_res = None
        self.alpha = None
        self.T = None

    # Initialize the parameters of the HMM
    maxIterations = 100 # Maximum number of iterations
    iters = 0 # Number of iterations
    oldLogProb = -100000 # Previous log probability

    def alphaPass(self):
        #compute alpha[0][i]
        c0 = 0
        for i in range(self.N-1):
            self.alpha[0][i] = self.pi[i] * self.B[i][self.O[0]]
            c0 = c0 + self.alpha[0][i]
            sum(self.alpha[0])
            i = i + 1

        #scale the alpha[0][i]
        c0 = 1/c0
        for i in range(self.N-1):
            self.alpha[0][i] = self.alpha[0][i] * c0
            i = i + 1

        #compute alpha[t][i]
        for t in range(1, self.T-1):
            ct = 0
            for i in range(self.N-1):
                self.alpha[t][i] = 0
                for j in range(self.N-1):
                    self.alpha[t][i] = self.alpha[t][i] + self.alpha[t-1][j] * self.A[j][i]
                    j = j + 1
                self.alpha[t][i] = self.alpha[t][i] * self.B[i][self.O[t]]
                ct = ct + self.alpha[t][i]
                i = i + 1

            #scale alpha[t][i]
            ct = 1/ct
            for i in range(self.N-1):
                self.alpha[t][i] = self.alpha[t][i] * ct
                i = i + 1

        return self.alpha

    def betaPass(self):
        #Let beta[T-1][i] = 1 scaled by c[T-1]
        ct = 1/self.c[self.T-1]
        for i in range(self.N-1):
            self.beta[self.T-1][i] = ct
            i = i + 1

        #beta pass
        for t in range(self.T-2, 0, -1):
            for i in range(self.N-1):
                self.beta[t][i] = 0
                for j in range(self.N-1):
                    self.beta[t][i] = self.beta[t][i] + self.A[i][j] * self.B[j][self.O[t+1]] * self.beta[t+1][j]
                    j = j + 1
                #scale beta[t][i] with same scale factor as alpha[t][i]
                self.beta[t][i] = self.beta[t][i] * self.c[t]
                i = i + 1

        return self.beta

    def computeGamma(self):
        denom = 0
        for i in range(self.N-1):
            for j in range(self.N-1):
                denom = denom + self.alpha[self.T-1][i] * self.beta[self.T-1][j]
                j = j + 1
            i = i + 1

        for t in range(self.T-1):
            for i in range(self.N-1):
                self.gamma[t][i] = 0
                for j in range(self.N-1):
                    self.gamma[t][i] = self.gamma[t][i] + self.alpha[t][i] * self.A[i][j] * self.B[j][self.O[t+1]] * self.beta[t+1][j]
                    j = j + 1
                self.gamma[t][i] = self.gamma[t][i] / denom
                i = i + 1

        return self.gamma

    def computeDiGamma(self):
        denom = 0
        for i in range(self.N-1):
            for j in range(self.N-1):
                denom = denom + self.alpha[self.T-1][i] * self.beta[self.T-1][j]
                j = j + 1
            i = i + 1

        for t in range(self.T-2):
            for i in range(self.N-1):
                for j in range(self.N-1):
                    self.diGamma[t][i][j] = (self.alpha[t][i] * self.A[i][j] * self.B[j][self.O[t+1]] * self.beta[t+1][j]) / denom
                    j = j + 1
                i = i + 1

        return self.diGamma

    def reEstimate(self):
        #re-estimate pi
        for i in range(self.N-1):
            self.pi[i] = self.gamma[0][i]
            i = i + 1

        #re-estimate A
        for i in range(self.N-1):
            for j in range(self.N-1):
                numer = 0
                denom = 0
                for t in range(self.T-2):
                    numer = numer + self.diGamma[t][i][j]
                    denom = denom + self.gamma[t][i]
                    t = t + 1
                self.A[i][j] = numer / denom
                j = j + 1
            i = i + 1

        #re-estimate B
        for i in range(self.N-1):
            for j in range(self.M-1):
                numer = 0
                denom = 0
                for t in range(self.T-1):
                    if self.O[t] == j:
                        numer = numer + self.gamma[t][i]
                    denom = denom + self.gamma[t][i]
                    t = t + 1
                self.B[i][j] = numer / denom
                j = j + 1
            i = i + 1

        return self.A, self.B, self.pi

    def computeLogProb(self):
        logProb = 0
        for i in range(self.T-1):
            logProb = logProb + math.log(self.c[i])
            i = i + 1
        return -logProb

    def train(self):
        self.alphaPass()
        self.betaPass()
        self.computeGamma()
        self.computeDiGamma()
        self.reEstimate()
        self.logProb = self.computeLogProb()
        self.iters = self.iters + 1

    def trainUntilConverge(self):
        oldLogProb = 0
        self.train()
        while self.logProb > oldLogProb:
            oldLogProb = self.logProb
            self.train()
        return self.A, self.B, self.pi
