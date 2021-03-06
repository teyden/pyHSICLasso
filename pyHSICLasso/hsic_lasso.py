#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import dict, range

from future import standard_library

import enum
import numpy as np
from joblib import Parallel, delayed

from .kernel_tools import kernel_delta_norm, kernel_gaussian, kernel_custom, get_phylogenetic_tree

standard_library.install_aliases()

class CustomKernel(enum.Enum): 
    Jaccard = "jaccard"
    BrayCurtis = "braycurtis"
    UnweightedUniFrac = "unweighted_unifrac"
    WeightedUniFrac = "weighted_unifrac"

def hsic_lasso(X, Y, y_kernel, x_kernel='Gaussian', zero_adjust=True, n_jobs=-1, discarded=0, B=0, M=1, featname=None):
    """
    Input:
        X      input_data
        Y      target_data
        y_kernel  We employ the Gaussian kernel for inputs. For output kernels,
                  we use the Gaussian kernel for regression cases and
                  the delta kernel for classification problems.
    Output:
        X         matrix of size d x (n * B (or n) * M)
        X_ty      vector of size d x 1
    """
    d, n = X.shape
    dy = Y.shape[0]

    # Computes the kernel for the outcome
    L = compute_kernel(Y, y_kernel, B, M, discarded)
    L = np.reshape(L, (n * B * M,1))

    # Construct the phylogenetic tree here once and pass it around
    tree, internal_ids, internal_to_otu_map, otu_to_internal_map = None, None, None, None
    if x_kernel == "UnweightedUniFrac" or x_kernel == "WeightedUniFrac":
        tree, internal_ids, internal_to_otu_map, otu_to_internal_map = get_phylogenetic_tree(features=featname)

    # Preparing design matrix for HSIC Lars
    # TODO - what is the output like?
    result = Parallel(n_jobs=n_jobs)([delayed(parallel_compute_kernel)(
        np.reshape(X[k, :], (1, n)), x_kernel, k, B, M, n, discarded, zero_adjust, featname, tree, otu_to_internal_map) for k in range(d)])

    # non-parallel version for debugging purposes
    # result = []
    # for k in range(d):
    #     X = parallel_compute_kernel(X[k, :], x_kernel, k, B, M, n, discarded)
    #     result.append(X)

    result = dict(result)

    # This is the flattened kernel "matrix" for the X input. It has n * B * M rows, and d number of feature columns.
    K = np.array([result[k] for k in range(d)]).T

    # This is the a dot product of the transposed kernel matrix for the X input and the kernel matrix for the outcome
    # What is it used for?
    KtL = np.dot(K.T, L)

    return K, KtL, L


def _compute_custom_kernel(x, kernel, zero_adjust=True, featname=None, feature_idx=None, tree=None, otu_to_internal_map=None):
    try:    
        _kernel = CustomKernel[kernel].value
    except:
        print("Kernel metric provided doesn't match valid options.")
    return kernel_custom(x, _kernel, zero_adjust, featname, feature_idx, tree, otu_to_internal_map)


def compute_kernel(x, kernel, B=0, M=1, discarded=0, zero_adjust=True, featname=None, feature_idx=None, tree=None, otu_to_internal_map=None):

    d,n = x.shape

    H = np.eye(B, dtype=np.float32) - 1 / B * np.ones(B, dtype=np.float32)
    K = np.zeros(n * B * M, dtype=np.float32) # Flattened matrix

    # Normalize data
    if kernel == "Gaussian":
        x = (x / (x.std() + 10e-20)).astype(np.float32)

    """
    This method goes through and computes blocks of kernels for single features, for random subsets of samples.
    """
    st = 0
    ed = B ** 2
    index = np.arange(n)
    for m in range(M):
        np.random.seed(m)
        index = np.random.permutation(index)

        # I believe the distance between i and j will always be B. This defines the block size to be computed.
        for i in range(0, n - discarded, B):
            j = min(n, i + B)
            
            # Grabs random columns between i and j of index, the random container of indices between 0:n
            # I think it computes the sample-sample differences for a single feature.
            block = x[:,index[i:j]]
            if kernel == 'Gaussian':
                k = kernel_gaussian(block, block, np.sqrt(d))
            elif kernel == 'Delta':
                k = kernel_delta_norm(block, block)
            # TODO test this; how is this k diff from the above?
            elif kernel in ["Jaccard", "BrayCurtis", "UnweightedUniFrac", "WeightedUniFrac"]:
                k = _compute_custom_kernel(
                    block.T, kernel, zero_adjust=zero_adjust, featname=featname, feature_idx=feature_idx, tree=tree, otu_to_internal_map=otu_to_internal_map)
            else:
                raise Exception("Invalid kernel selection.")

            k = np.dot(np.dot(H, k), H)

            # Normalize HSIC tr(k*k) = 1
            k = k / (np.linalg.norm(k, 'fro') + 10e-10)
            K[st:ed] = k.flatten() # This sets the columns for K from range "st" up to "ed"
            st += B ** 2
            ed += B ** 2

    return K


def parallel_compute_kernel(x, kernel, feature_idx, B, M, n, discarded, zero_adjust, featname, tree, otu_to_internal_map):
    return (feature_idx, compute_kernel(x, kernel, B, M, discarded, zero_adjust, featname, feature_idx, tree, otu_to_internal_map))
