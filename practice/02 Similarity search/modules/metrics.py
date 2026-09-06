import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    ed_dist = 0

    # INSERT YOUR CODE

    return ed_dist

def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    norm_ed_dist = 0

    # INSERT YOUR CODE

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance with Sakoe-Chiba band constraint

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window radius (0 <= r <= n-1)

    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """
    ts1 = np.asarray(ts1, dtype=float)
    ts2 = np.asarray(ts2, dtype=float)
    n, m = len(ts1), len(ts2)

    if n == 0 or m == 0:
        return float("inf")

    window = int(r * n)

    INF = np.inf
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            diff = ts1[i - 1] - ts2[j - 1]
            cost = np.sum(diff * diff)
            D[i, j] = cost + min(D[i - 1, j],
                                 D[i, j - 1],
                                 D[i - 1, j - 1])

    return float(D[n, m])



