import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall


def pairWise2Transitive(data):
    """
    Given a binary array `data`, returns a transitive closure of the pairs.
    Specifically, if A and B are elements in `data` such that A and B are both 1,
    or A is 1 and there exists some element C such that C and B are both 1, then
    the corresponding element in the returned array is 1. Similarly, if A and B
    are both -1 or there exists some element C such that C is -1 and either A and B
    are both 1 or there exists some element D such that D is 1 and both D and B are -1,
    then the corresponding element in the returned array is -1.

    Parameters:
        data (numpy.ndarray): Binary array of shape (n, n).

    Returns:
        numpy.ndarray: Transitive closure of `data` with shape (n, n).
    """
    positive_pairs = (data == 1) * 0.01
    pos_dist_matrix = floyd_warshall(csgraph=csr_matrix(positive_pairs), directed=False)

    negative_pairs = (data == -1) * 100.
    negative_pairs = positive_pairs + negative_pairs
    neg_dist_matrix = floyd_warshall(csgraph=csr_matrix(negative_pairs), directed=False)

    transitive = np.zeros_like(data)
    transitive[np.all([100 <= neg_dist_matrix, neg_dist_matrix < 200], axis=0)] = -1
    transitive[pos_dist_matrix < np.inf] = 1

    return transitive


def countTransitivePairs(data):
    """
    Given a binary array `data`, returns the number of transitive pairs in `data`.
    Specifically, returns the number of pairs that are transitive and have a positive value
    as well as the number of pairs that are transitive and have a negative value.

    Parameters:
        data (numpy.ndarray): Binary array of shape (n, n).

    Returns:
        Tuple[int, int]: A tuple of two integers representing the number of transitive pairs
        with positive and negative values respectively.
    """
    transitive = pairWise2Transitive(data)
    tril = np.tril(transitive, -1)
    return tril[tril > 0].sum(), tril[tril < 0].sum()
