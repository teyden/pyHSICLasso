#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from future import standard_library

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances

from skbio.diversity.beta import unweighted_unifrac

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

#####################################################################################

def kernel_custom(X, kernel, zero_adjust=False, featname=None):
    if zero_adjust:
        """
        Cite: https://github.com/phytomosaic/ecole/blob/master/R/bray0.R
        The idea of adding a pseudo species is to account for cases where sample units do not have
        any shared species at all. This will cause for Bray-Curtis dissimilarity to saturate at 1,
        providing little information about the true 'dissimilarity' of no-share sample unit pairs.
        This will mean that for those with no shared taxa, this pseudo column will be the single
        shared taxa between the sample. Instead of saturating at 1, it will saturate at a high
        value close to 1. BC actually produces NaN if two samples have no shared units. 
        Hence, it is possible to have dissimilarity=1 with shared units between samples.
        I thought that dissimilarity could only be 1 if there are no shared units, but this is
        potentially incorrect.  
        Consider adding this pseudo column to the original X matrix instead, prior to modelling.
        If this is done here for HSIC, then the same should be done for RFECV. 

        Tricky fact: if adding a pseudo species with a min count for a "block" vs. the entire X matrix,
        then it may result in weird results....
        I'd likely only use this when working with the whole X matrix.

        :param X: an OTU table with samples as rows and OTUs as columns
        :return:
        """
        X = add_pseudo_species(X)

    if kernel == "jaccard":
        # Temporarily match the Jaccard computation with the vegan::vegdist implementation in R.
        D = pairwise_distances(X, metric="braycurtis")
        D = (2 * D) / (1 + D)
    if kernel == "unweighted_unifrac":
        D = pw_dist_unifrac(X, featname)
    else:
        # TODO - unifrac can be implemented using beta_diversity.unweighted_unifrac
        # it creates a distance matrix
        # need to construct a tree: http://scikit-bio.org/docs/0.4.2/generated/generated/skbio.diversity.beta.unweighted_unifrac.html
        D = pairwise_distances(X, metric=kernel)

    # For samples with no shared taxa, NaN would be produced. Replace it with 1.
    # Will saturate at 1. Could have side effects, but this must be matched with the
    # RFECV implementation. 
    D[np.isnan(D)] = 1
    K = convert_D_to_K(D)

    return K

from skbio.diversity import beta_diversity
from skbio.tree import TreeNode
from io import StringIO

def pw_dist_unifrac(X, featname):
    fp_microbiome_data = "/Users/teyden/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__18__3m-ds5y_microbiome__train.csv"
    fp_taxa_mapping = "/Users/teyden/Projects/asthma/data/child-study-data-jan-2019/processed/microbiome-data/taxonomy_table.csv"
    fp_tree = "/Users/teyden/Projects/asthma/data/child-study-data-jan-2019/their-files/tree.nwk"
    
    with open(fp_tree, 'r') as file:
        newick_tree = file.read()
    tree = TreeNode.read(StringIO(newick_tree))
    tree = tree.root_at("root")

    INTERNALID_TO_OTUID_MAPPING = {}
    OTUID_TO_INTERNALID_MAPPING = {}
    count = 0
    for idx, node in tree.to_array()["id_index"].items():
        if node.is_tip():
            count += 1
            INTERNALID_TO_OTUID_MAPPING[node.name] = ""
        
    for row in taxa_mapping.iterrows():
        internal_id = row[1].internal_id
        otu_id = row[1].otu_identifier
        if internal_id in ids_dict:
            INTERNALID_TO_OTUID_MAPPING[internal_id] = otu_id
        else:
            print("This OTU is not in the tree: ", internal_id)
            
    for internal_id, otu_id in INTERNALID_TO_OTUID_MAPPING.items():
        OTUID_TO_INTERNALID_MAPPING[otu_id] = internal_id
    
    internal_ids = get_internal_ids(featname, mapping=OTUID_TO_INTERNALID_MAPPING)
    uw_u_D = beta_diversity("unweighted_unifrac", counts=X, tree=tree, otu_ids=internal_ids)
    return uw_u_D.data

def get_internal_ids(otu_ids, mapping):
    internal_ids = []
    for otu_id in otu_ids:
        if otu_id in mapping:
            internal_ids.append(mapping.get(otu_id))
        else:
            raise Exception("OTU ID provided has no internal ID mapping")
    return internal_ids
    
def convert_D_to_K(D):
    """
    Applies positive-semidefinite correction to project the distance matrix to the kernel space.
    :param D: a square distance matrix
    :return: a square kernel matrix
    """
    n, d = D.shape
    if n != d:
        raise ValueError("Expects a square distance matrix for conversion to a kernel. " +
                         "Handling for rectangular matrix not yet supported.")
    n_I = np.array([1] * n)
    center = np.diag(n_I) - 1 / n
    # if n != d:
    #     K = -0.5 * (center @ (D * D)).T @ center
    K = -0.5 * center @ (D * D) @ center
    u, s, vh = np.linalg.svd(K)
    K = u @ np.diag(np.maximum(np.zeros(n), s)) @ vh
    return K

def find_nonzero_min(X):
    min_val = max(0, np.min(X[X != 0]))
    return min_val

def add_pseudo_species(X, min_val=None):
    """
    Adds a pseudo column to the X matrix taking the minimum non-zero value of the entire matrix.
    :param X: assumes samples are rows and features are columns
    :return: The augmented matrix.
    """
    if min_val is None:
        min_val = find_nonzero_min(X)
        if min_val == 0:
            raise ValueError("This OTU matrix contains all zeros.")

    # This is an nx1 array that looks like: np.array([[1],[2],[3]]). 
    if X.shape[1] == 1: 
        X = np.append(X, np.array([[min_val]]), axis=0)
    else:
        n, d = X.shape
        scaffold = np.zeros((n, d + 1)) + min_val
        scaffold[:, :-1] = X
        X = scaffold
    return X






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