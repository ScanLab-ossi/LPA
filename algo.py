import numpy as np
import numpy.ma as ma


def over_under(P, Q):
    """np.where(P < Q, -1, 1) adds a minus sign if P < Q"""
    return np.where(P < Q, -1, 1)


def KLD_distance(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Kullback-Leibler distance normalized so as to be a distance complying to the triangle inequality.
    P represents the data, the observations, or a measured probability distribution.
    Q represents a theory, a model, a description or an approximation of P.
    P - Q makes the regular KL divergence a distance complying with the triangle inequality.
    The distance is always > 0.
    """
    return (P - Q) * np.log2(np.where(P == 0, Q, P) / Q)


# def KLD_distance_overused(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
#     """
#     Kullback-Leibler distance normalized so as to be a distance complying to the triangle inequality.
#     P represents the data, the observations, or a measured probability distribution.
#     Q represents a theory, a model, a description or an approximation of P.
#     P - Q makes the regular KL divergence a distance complying with the triangle inequality.
#     `over_under(P,Q)` adds a minus sign if P < Q
#     """
#     return over_under(P, Q) * (P - Q) * np.log2(np.where(P == 0, Q, P) / Q)


def KL_divergence(P, Q):
    """Kullback-Leibler divergence. Note that the np.where is so that the divergence == 0 when in the limit P -> 0."""
    return P * np.log2(np.where(P == 0, Q, P) / Q)


def symmetrized_KLD(P, Q):
    """A symmetrized Kullback-Leibler divergence.
    See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence.
    P represents the data, the observations, or a measured probability distribution.
    Q represents a theory, a model, a description or an approximation of P.
    P - Q makes the regular KL divergence a distance complying with the triangle inequality.
    `over_under(P,Q)` adds a minus sign if P < Q"""
    return over_under(P, Q) * KLD_distance(P, Q)


def entropy(P: np.ndarray) -> np.ndarray:
    """
    Shannon entropy without summation.
    """
    P = np.multiply(P, np.log2(P, where=P > 1e-10))
    return np.where(P != 0, -P, P)


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


# def KLD_distance_overused(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
#     """
#     Kullback-Leibler distance normalized so as to be a distance complying to the triangle inequality.
#     P represents the data, the observations, or a measured probability distribution.
#     Q represents a theory, a model, a description or an approximation of P.
#     P - Q makes the regular KL divergence a distance complying with the triangle inequality.
#     np.where(P < Q, -1, 1) adds a minus sign if P < Q
#     """
#     return (P - Q) * np.log2(np.where(P == 0, Q, P) / Q)

# arr = np.subtract(P, Q)
# arr2 = np.divide(P, Q)
# np.log2(arr2, out=arr2)
# np.multiply(arr, arr2, out=arr)
# np.multiply(np.where(P < Q, -1, 1), arr, out=arr)
# return arr
