import sys
import math

sys.setrecursionlimit(100000)

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
        self.forward_res = None
        self.viterbi_res = None
        self.alpha = None

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
            for j in range(self.M):
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
                break
            log_p = self.prob_log(p)
        
    def printer(self):
        # Convert self.A to a string
        A = ""
        for i in range(self.N):
            for j in range(self.N):
                A += str(round(self.A[i][j], 6)) + " "
        
        # Convert self.B to a string
        B = ""
        for i in range(self.N):
            for j in range(self.M):
                B += str(round(self.B[i][j], 6)) + " "
        
        sys.stdout.write(str(self.N) + " " + str(self.N) + " " + A + "\n")
        sys.stdout.write(str(self.N) + " " + str(self.M) + " " + B)

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

    def clear_everything(self):
        self.A = None
        self.B = None
        self.pi = None
        self.O = None
        self.forward_res = None
        self.viterbi_res = None

def main():
    model = HMM()

    # Read the input
    global A, B, pi, O
    input = sys.stdin.read().splitlines()
    transition_matrix = input[0].split()
    transition_matrix = [float(i) for i in transition_matrix]
    t_shape = (int(transition_matrix[0]), int(transition_matrix[1]))
    transition_matrix = transition_matrix[2:]
    transition_matrix = make_matrix(transition_matrix, t_shape)
    model.A = transition_matrix

    emission_matrix = input[1].split()
    emission_matrix = [float(i) for i in emission_matrix]
    e_shape = (int(emission_matrix[0]), int(emission_matrix[1]))
    emission_matrix = emission_matrix[2:]
    emission_matrix = make_matrix(emission_matrix, e_shape)
    model.B = emission_matrix
    
    initial_matrix = input[2].split()
    initial_matrix = [float(i) for i in initial_matrix]
    i_shape = (int(initial_matrix[0]), int(initial_matrix[1]))
    initial_matrix = initial_matrix[2:]
    initial_matrix = make_matrix(initial_matrix, i_shape)
    model.pi = initial_matrix

    observation_sequence = input[3].split()
    observation_sequence = [int(i) for i in observation_sequence]
    observation_sequence = observation_sequence[1:]
    model.O = observation_sequence

    #initial_delta_idx = [[None] * len(model.pi[0])]
    # #model.viterbi(initial_alpha, initial_delta_idx, model.O[1:])

    model.M = len(set(model.O))
    model.N = len(model.A)
    model.T = len(model.O)
    model.baum_welch(50)
    model.printer()


if __name__ == "__main__":
    main()