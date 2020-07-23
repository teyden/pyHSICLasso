from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from future import standard_library

from pyHSICLasso import HSICLasso

import scipy.io as sio
import pandas as pd 
import numpy as np

standard_library.install_aliases()


def main():

    #Numpy array input example
    hsic_lasso = HSICLasso()

    fp_microbiome_data = "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__18__3m-ds5y_microbiome__train.csv"
    fp_metadata = "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__18__3m-ds5y_metadata__train.csv"

    mb = pd.read_csv(fp_microbiome_data, index_col=0)
    metadata = pd.read_csv(fp_metadata, index_col=0)
    
    OTU_IDs = mb.columns.values
    X = np.array(mb)
    Y = np.array(metadata["diseasestatus_5y_binary"])
    d = mb.shape[1] - 1

    hsic_lasso.input(X, Y, featname=OTU_IDs)
    hsic_lasso.classification(d,
                              y_kernel="Delta",
                              x_kernel="UnweightedUniFrac",
                              covars_kernel="Gaussian",
                              zero_adjust=False)
    hsic_lasso.dump()
    hsic_lasso.plot_path()

    #Save parameters
    hsic_lasso.save_param()

if __name__ == "__main__":
    main()
