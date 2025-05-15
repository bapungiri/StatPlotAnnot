import numpy as np
from scipy.stats import bootstrap


def get_bootstrap_prob(sample1, sample2):
    """
    get_direct_prob Returns the direct probability of items from sample2 being
    greater than or equal to those from sample1.
       Sample1 and Sample2 are two bootstrapped samples and this function
       directly computes the probability of items from sample 2 being greater
       than or equal to those from sample1. Since the bootstrapped samples are
       themselves posterior distributions, this is a way of computing a
       Bayesian probability. The joint matrix can also be returned to compute
       directly upon.
    obtained from: https://github.com/soberlab/Hierarchical-Bootstrap-Paper/blob/master/Bootstrap%20Paper%20Simulation%20Figure%20Codes.ipynb

    References
    ----------
    Saravanan, Varun, Gordon J Berman, and Samuel J Sober. “Application of the Hierarchical Bootstrap to Multi-Level Data in Neuroscience.” Neurons, Behavior, Data Analysis and Theory 3, no. 5 (2020): https://nbdt.scholasticahq.com/article/13927-application-of-the-hierarchical-bootstrap-to-multi-level-data-in-neuroscience.

    Parameters
    ----------
    sample1: array
        numpy array of values
    sample2: array
        numpy array of values

    Returns
    ---------
    pvalue1
        using joint probability distribution
    pvalue2
        using the number of sample2 being greater than sample1
    2d array
        joint probability matrix
    """

    # assert len(sample1) == len(sample2), "both inputs lengths should match"
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)

    joint_low_val = min([min(sample1), min(sample2)])
    joint_high_val = max([max(sample1), max(sample2)])

    nbins = 100
    p_axis = np.linspace(joint_low_val, joint_high_val, num=nbins)
    edge_shift = (p_axis[2] - p_axis[1]) / 2
    p_axis_edges = p_axis - edge_shift
    p_axis_edges = np.append(p_axis_edges, (joint_high_val + edge_shift))

    # Calculate probabilities using histcounts for edges.

    p_sample1 = np.histogram(sample1, bins=p_axis_edges)[0] / np.size(sample1)
    p_sample2 = np.histogram(sample2, bins=p_axis_edges)[0] / np.size(sample2)

    # Now, calculate the joint probability matrix:
    # p_joint_matrix = np.zeros((nbins, nbins))
    # for i in np.arange(np.shape(p_joint_matrix)[0]):
    #     for j in np.arange(np.shape(p_joint_matrix)[1]):
    #         p_joint_matrix[i, j] = p_sample1[i] * p_sample2[j]

    p_joint_matrix = p_sample1[:, np.newaxis] * p_sample2[np.newaxis, :]

    # Normalize the joint probability matrix:
    p_joint_matrix = p_joint_matrix / p_joint_matrix.sum()

    # Get the volume of the joint probability matrix in the upper triangle:
    p_test = np.sum(np.triu(p_joint_matrix))
    p_test = 1 - p_test if p_test >= 0.5 else p_test

    statistic = np.abs(sample1.mean() - sample2.mean()) / np.sqrt(
        (sample1.std() ** 2 + sample2.std() ** 2) / 2
    )
    return statistic, np.abs(p_test)


def get_bootstrap_prob_paired(arr1, arr2):
    l1, l2 = len(arr1), len(arr2)
    assert l1 == l2, f"len(arr1)={l1},len(arr2)={l2}: both inputs lengths should match"
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    delta = arr1 - arr2
    p_test = np.sum(delta >= 0) / len(delta)
    p_test = 1 - p_test if p_test >= 0.5 else p_test

    return np.nanmean(delta), p_test


def bootstrap_test(x, y, statistic, n_resamples=10000, paired=False):

    # Define statistic function for bootstrap — must take a tuple of arrays
    def fun(x, y):
        return statistic(x), statistic(y)

    x_stat_samples, y_stat_samples = bootstrap(
        (x, y), statistic=fun, n_resamples=n_resamples, paired=paired, vectorized=False
    ).bootstrap_distribution

    if paired:
        stat, p_value = get_bootstrap_prob_paired(x_stat_samples, y_stat_samples)
    else:
        stat, p_value = get_bootstrap_prob(x_stat_samples, y_stat_samples)

    return stat, p_value
