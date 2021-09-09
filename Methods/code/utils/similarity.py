import numpy as np


def get_Jaccard_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    E = np.ones_like(X.T)
    denominator=X * E + E.T * X.T - X * X.T
    denominator_zero_index=np.where(denominator==0)
    denominator[denominator_zero_index]=1
    result = X * X.T / denominator
    result[denominator_zero_index]=0
    result = result - np.diag(np.diag(result))
    return result


def get_CommonNeighbours_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    similarity_matrix = X * X.T
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    return matrix_normalize(similarity_matrix)


def get_Salton_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    alpha = np.multiply(X, X).sum(axis=1)
    similarity_matrix = X * X.T / (np.sqrt(alpha * alpha.T))
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    return matrix_normalize(similarity_matrix)


def get_Sorensen_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    alpha = X.sum(axis=1)
    similarity_matrix = 2 * X * X.T / (alpha + alpha.T)
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    return matrix_normalize(similarity_matrix)


def get_Hub_Promoted_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    alpha = X.sum(axis=1)
    similarity_matrix = 2 * X * X.T / (alpha + alpha.T - abs(alpha - alpha.T))
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    return matrix_normalize(similarity_matrix)


def get_Hub_Depressed_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    alpha = X.sum(axis=1)
    similarity_matrix = 2 * X * X.T / (alpha + alpha.T + abs(alpha - alpha.T))
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    return matrix_normalize(similarity_matrix)


def get_LHN1_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    alpha = X.sum(axis=1)
    similarity_matrix = X * X.T / (alpha * alpha.T)
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    return matrix_normalize(similarity_matrix)


def get_PA_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    alpha = X.sum(axis=1)
    similarity_matrix = alpha * alpha.T
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    return matrix_normalize(similarity_matrix)


def get_AA_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    alpha = np.log(X.sum(axis=0))
    alpha[np.where(alpha == 0)] = 1
    alpha = 1 / alpha
    result = np.multiply(X, alpha) * X.T
    result = result - np.diag(np.diag(result))
    similarity_matrix = result
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    return matrix_normalize(similarity_matrix)


def get_RA_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    alpha = 1 / X.sum(axis=0)
    result = np.multiply(X, alpha) * X.T
    result = result - np.diag(np.diag(result))
    similarity_matrix = result
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    return matrix_normalize(similarity_matrix)


def get_Cosin_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    alpha = np.multiply(X, X).sum(axis=1)
    #print(np.where(alpha==0))
    norm=alpha*alpha.T
    index=np.where(norm== 0)
    norm[index]=1
    similarity_matrix = X * X.T / (np.sqrt(norm))
    similarity_matrix[index]=0
    #similarity_matrix[np.isnan(similarity_matrix)] = 0
    result=similarity_matrix
    result = result - np.diag(np.diag(result))
    return result


def get_Pearson_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    X = X - (X.sum(axis=1) / X.shape[1])
    similarity_matrix = get_Cosin_Similarity(X)
    similarity_matrix[np.isnan(similarity_matrix)] = 0

    return similarity_matrix


def get_Katz_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    eigenvalue, eigenvector = np.linalg.eig(X)
    eigenvalue = np.max(eigenvalue)
    alpha = 0.5 * (1 / eigenvalue)
    similarity_matrix = np.linalg.inv(np.identity(len(X)) - alpha * X) - np.identity(len(X))
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    return matrix_normalize(similarity_matrix)


def get_Gauss_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    delta = 1 / np.mean(np.power(X,2), 0).sum()
    alpha = np.power(X, 2).sum(axis=1)
    result = np.exp(np.multiply(-delta, alpha + alpha.T - 2 * X * X.T))
    #similarity_matrix[np.isnan(similarity_matrix)] = 0
    result = result - np.diag(np.diag(result))
    return result


def matrix_normalize(similarity_matrix):
    similarity_matrix[np.isnan(similarity_matrix)] = 0
    if similarity_matrix.shape[0] == similarity_matrix.shape[1]:
        for i in range(similarity_matrix.shape[0]):
            similarity_matrix[i, i] = 0
        for i in range(200):
            D = np.diag(np.array(np.sum(similarity_matrix, axis=1)).flatten())
            D = np.linalg.pinv(np.sqrt(D))
            similarity_matrix = D * similarity_matrix * D
    else:
        for i in range(similarity_matrix.shape[0]):
            if np.sum(similarity_matrix[i], axis=1) == 0:
                similarity_matrix[i] = similarity_matrix[i]
            else:
                similarity_matrix[i] = similarity_matrix[i] / np.sum(similarity_matrix[i], axis=1)
    return similarity_matrix


def fast_calculate(feature_matrix, neighbor_num):
    iteration_max = 50
    mu = 6
    X = feature_matrix
    alpha = np.power(X, 2).sum(axis=1)
    distance_matrix=np.sqrt(alpha+alpha.T-2*X*X.T)
    row_num = X.shape[0]
    e=np.ones((row_num,1))
    distance_matrix=np.array(distance_matrix+np.diag(np.diag(e*e.T*np.inf)))
    nearest_neighbor_matrix=np.zeros((row_num,row_num))
    sort_index = np.argsort(distance_matrix,kind='mergesort')
    nearest_index = sort_index[:, :neighbor_num].flatten()
    nearest_neighbor_matrix[np.arange(row_num).repeat(neighbor_num), nearest_index] = 1
    C=nearest_neighbor_matrix
    np.random.seed(0)
    W= np.mat(np.random.rand(row_num,row_num), dtype=float)
    W=np.multiply(C,W)
    lamda=mu*e
    P=X*X.T+lamda*e.T
    for q in range(iteration_max):
        Q=W*P
        W=np.multiply(W,P)/Q
        W=np.nan_to_num(W)
    return W

