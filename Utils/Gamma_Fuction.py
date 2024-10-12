import numpy as np
import scipy.special as sc

# Gamma Fuction
def gamma_function(x):
    return sc.gamma(x + 1)    # Gamma(n): I(n) = (n - 1)! 

# Fuzzy Rank Sum (FRS)
def calculate_frs(confidence_scores, gamma_shape):
    ranked_scores = [gamma_function(score) for score in confidence_scores]
    # print('sum',np.sum(ranked_scores, axis=1))
    return np.sum(ranked_scores, axis=1)

# Complement of the Confidence Factor Sum (CCFS)
def calculate_ccfs(confidence_scores):
    complement_scores = [1 - score for score in confidence_scores]
    ccfs = np.mean(complement_scores, axis=1)
    return ccfs

# Final Decision Score (FDS)
def calculate_fds(frs, ccfs):
    return frs * ccfs

def gamma_function_ranking(confidence_scores, gamma_shape=1.0, M=3, C=1):

    # confidence_scores = {
    #     'class': [0.81, 0.96, 0.94],
    # }

    # Select the shape parameter of the Gamma function, following the formula in the documentation
    # here we use the default value 1.0   
    confidence_scores = np.stack([t.detach().cpu().numpy() for t in confidence_scores], axis=0)
    # print(confidence_scores)
    FRS = calculate_frs(confidence_scores, gamma_shape)
    CCFS = calculate_ccfs(confidence_scores)

    FDS = calculate_fds(FRS, CCFS)
    # print('FDS', FDS,'CCFS', CCFS,'FRS',FRS)
    max_fds_index = np.argmin(FDS)

    max_fds = confidence_scores[max_fds_index]
    return max_fds.mean()
    # print(f"Fuzzy Rank Sum (FRS) for Benign: {FRS}")
    # print(f"Complement of the Confidence Factor Sum (CCFS) for Benign: {CCFS}")
    # print(f"Final Decision Score (FDS) for Benign: {FDS}")