#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import dict, range

from future import standard_library

import enum
import numpy as np
from joblib import Parallel, delayed

from .kernel_tools import kernel_delta_norm, kernel_gaussian, kernel_custom

standard_library.install_aliases()

class CustomKernel(enum.Enum): 
    Jaccard = "jaccard"
    BrayCurtis = "braycurtis"
    UnweightedUniFrac = "unweighted_unifrac"
    WeightedUniFrac = "weighted_unifrac"

def hsic_lasso(X, Y, y_kernel, x_kernel='Gaussian', n_jobs=-1, discarded=0, B=0, M=1):
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
    L = np.reshape(L,(n * B * M,1))

    # Preparing design matrix for HSIC Lars
    # TODO - what is the output like?
    result = Parallel(n_jobs=n_jobs)([delayed(parallel_compute_kernel)(
        np.reshape(X[k,:],(1,n)), x_kernel, k, B, M, n, discarded) for k in range(d)])

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

def _compute_custom_kernel(x, kernel):
    try:    
        _kernel = CustomKernel[kernel].value
    except:
        print("Kernel metric provided doesn't match valid options.")
    return kernel_custom(x, _kernel)

def compute_kernel(x, kernel, B = 0, M = 1, discarded = 0):

    d,n = x.shape

    H = np.eye(B, dtype=np.float32) - 1 / B * np.ones(B, dtype=np.float32)
    K = np.zeros(n * B * M, dtype=np.float32) # Flattened matrix

    # Normalize data
    if kernel == "Gaussian":
        x = (x / (x.std() + 10e-20)).astype(np.float32)

    st = 0
    ed = B ** 2
    index = np.arange(n)
    for m in range(M):
        np.random.seed(m)
        index = np.random.permutation(index)

        # I believe the distance between i and j will always be B. This defines the block size to be computed.
        for i in range(0, n - discarded, B):
            j = min(n, i + B)

            block = x[:,index[i:j]] # All rows (the outcome variables), just columns i to j (the samples)
            if kernel == 'Gaussian':
                k = kernel_gaussian(block, block, np.sqrt(d))
            elif kernel == 'Delta':
                k = kernel_delta_norm(block, block)
            elif kernel in ["Jaccard", "Bray-Curtis"]:  # TODO test this; how is this k diff from the above?
                k = _compute_custom_kernel(block)

            k = np.dot(np.dot(H, k), H)

            # Normalize HSIC tr(k*k) = 1
            k = k / (np.linalg.norm(k, 'fro') + 10e-10)
            K[st:ed] = k.flatten() # This sets the columns for K from range "st" up to "ed"
            st += B ** 2
            ed += B ** 2

    return K

def parallel_compute_kernel(x, kernel, feature_idx, B, M, n, discarded):

    return (feature_idx, compute_kernel(x, kernel, B, M, discarded))