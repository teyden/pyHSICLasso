#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from future import standard_library

from scipy.spatial.distance import pdist, squareform
import numpy as np

standard_library.install_aliases()

def kernel_delta_norm(X_in_1, X_in_2):
    # Since X_in_1 and 2 are the same, there should be lots of zeroes from kernel computation when computing
    n_1 = X_in_1.shape[1]
    n_2 = X_in_2.shape[1]
    K = np.zeros((n_1, n_2))
    u_list = np.unique(X_in_1)
    for ind in u_list:
        c_1 = np.sqrt(np.sum(X_in_1 == ind))
        c_2 = np.sqrt(np.sum(X_in_2 == ind))
        ind_1 = np.where(X_in_1 == ind)[1]
        ind_2 = np.where(X_in_2 == ind)[1]
        K[np.ix_(ind_1, ind_2)] = 1 / c_1 / c_2
    return K


def kernel_delta(X_in_1, X_in_2):
    n_1 = X_in_1.shape[1]
    n_2 = X_in_2.shape[1]
    K = np.zeros((n_1, n_2))

    # Get unique class labels
    u_list = np.unique(X_in_1)

    for ind in u_list:
        ind_1 = np.where(X_in_1 == ind)[1]
        ind_2 = np.where(X_in_2 == ind)[1]
        K[np.ix_(ind_1, ind_2)] = 1
    return K


def kernel_gaussian(X_in_1, X_in_2, sigma):
    n_1 = X_in_1.shape[1]
    n_2 = X_in_2.shape[1]
    X_in_12 = np.sum(np.power(X_in_1, 2), 0)
    X_in_22 = np.sum(np.power(X_in_2, 2), 0)
    dist_2 = np.tile(X_in_22, (n_1, 1)) + \
        np.tile(X_in_12, (n_2, 1)).transpose() - 2 * np.dot(X_in_1.T, X_in_2)
    K = np.exp(-dist_2 / (2 * np.power(sigma, 2)))
    return K

def kernel_custom(X, kernel):
    pairwise_distance_matrix = squareform(pdist(X, kernel))
    print(X)
    print(pairwise_distance_matrix)
    # TODO apply positive semidefinite correction? 
    return pairwise_distance_matrix

# data = [[23, 64, 14, 0, 0, 3, 1],
#          [0, 3, 35, 42, 0, 12, 1],
#          [0, 5, 5, 0, 40, 40, 0],
#          [44, 35, 9, 0, 1, 0, 0],
#          [0, 2, 8, 0, 35, 45, 1],
#          [0, 0, 25, 35, 0, 19, 0]]
#
# # Test whether kernel computation is consistent between this implementation and using a library
# from scipy.spatial.distance import pdist, squareform
#
# x = np.arange(5).reshape((1, 5)).T
#
# d = x.T.shape[0]
# s = np.sqrt(d)
#
# pairwise_dists_gaussian = squareform(pdist(x, 'sqeuclidean'))
# K = np.exp(-pairwise_dists_gaussian / (2 * np.power(s, 2)))
# _K = kernel_gaussian(x.T, x.T, s)
#
# # are K and _k equivalent? yes.
# # Now, do the same with Jaccard, but have to figure out how to convert it to a kernel.
# # Does it need to be corrected somehow? Is there an assumption that it comes from a reproducing kernel hilbert space?
#
#
#pairwise_dists_jaccard = squareform(pdist(x, 'jaccard'))
# # pairwise_dists_jaccard = skbio.diversity.beta_diversity(metric=kernel, counts=x)
#
# pairwise_dists_braycurtis = squareform(pdist(x, 'braycurtis'))