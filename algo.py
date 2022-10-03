import numpy as np


def KLD_distance(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Kullback-Leibler distance normalized so as to be a distance complying to the triangle inequality.
    P represents the data, the observations, or a measured probability distribution.
    Q represents a theory, a model, a description or an approximation of P.
    P - Q makes the regular KL divergence a distance complying with the triangle inequality.
    The distance is always > 0.
    """
    return (P - Q) * (np.log2(P) - np.log2(Q))


def KLD_distance_overused(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Kullback-Leibler distance normalized so as to be a distance complying to the triangle inequality.
    P represents the data, the observations, or a measured probability distribution.
    Q represents a theory, a model, a description or an approximation of P.
    P - Q makes the regular KL divergence a distance complying with the triangle inequality.
    np.where(P < Q, -1, 1) adds a minus sign if P < Q
    """
    return np.where(P < Q, -1, 1) * (P - Q) * (np.log2(P / Q))


def KLD_distance_consecutive(x: np.ndarray) -> np.ndarray:
    """
    Kullback-Leibler distance between a set of consecutive observations.
    """
    Q, P = x[:-1], x[1:]
    return (P - Q) * (np.log2(P) - np.log2(Q))


def KLD_divergence_consecutive(x: np.ndarray) -> np.ndarray:
    """
    Kullback-Leibler divergence between a set of consecutive observations.
    """
    Q, P = x[:-1], x[1:]
    return np.sum(P * (np.log2(P) - np.log2(Q)))


def KLD_divergence(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Kullback-Leibler divergence between a set of consecutive observations.
    """
    return np.sum(P * (np.log2(P) - np.log2(Q)))


def entropy(P: np.ndarray) -> np.ndarray:
    """
    Shannon entropy without summation.
    """
    return -P * np.log2(P)


def JSD(P_t: np.array) -> float:
    a = 1 / len(P_t)
    M = a * P_t.sum(axis=0)
    return a * np.sum([KLD_divergence(p, M) for p in P_t])


def JSD_max(P_t: np.array) -> float:
    a = 1 / len(P_t)
    M = a * P_t.sum(axis=0)
    return max([KLD_divergence(p, M) for p in P_t])


def sqrt_JSD(P_t) -> float:
    return np.sqrt(JSD(P_t))
