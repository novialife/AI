import sys

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


def make_matrix(matrix, shape):
    m = []
    for i in range(shape[0]):
        m.append(matrix[i*shape[1]:(i+1)*shape[1]])
    return m


def main():
    # Read the input
    input = sys.stdin.read().splitlines()
    transition_matrix = input[0].split()
    transition_matrix = [float(i) for i in transition_matrix]
    t_shape = (int(transition_matrix[0]), int(transition_matrix[1]))
    transition_matrix = transition_matrix[2:]
    transition_matrix = make_matrix(transition_matrix, t_shape)
    
    emission_matrix = input[1].split()
    emission_matrix = [float(i) for i in emission_matrix]
    e_shape = (int(emission_matrix[0]), int(emission_matrix[1]))
    emission_matrix = emission_matrix[2:]
    emission_matrix = make_matrix(emission_matrix, e_shape)
    
    initial_matrix = input[2].split()
    initial_matrix = [float(i) for i in initial_matrix]
    i_shape = (int(initial_matrix[0]), int(initial_matrix[1]))
    initial_matrix = initial_matrix[2:]
    initial_matrix = make_matrix(initial_matrix, i_shape)


    _ = multiply_matrices(initial_matrix, transition_matrix)
    res = multiply_matrices(_, emission_matrix)

    sys.stdout.write(str(len(res)))
    sys.stdout.write(" ")
    sys.stdout.write(str(len(res[0])))
    sys.stdout.write(" ")
    for i in range(len(res)):
        for j in range(len(res[i])):
            sys.stdout.write(str(res[i][j]))
            sys.stdout.write(" ")


if __name__ == "__main__":
    main()