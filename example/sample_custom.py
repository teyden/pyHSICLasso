import pandas as pd
import numpy as np 

from pyHSICLasso import HSICLasso

file_path_mb_data = "/Users/teyden/Projects/asthma/data/child-study-data-jan-2019/processed/microbiome-data/filtered_ra.3m.csv"
file_path_metadata = "~/Projects/asthma/data/child-study-data-jan-2019/processed/metadata/merged-metadata/merged_metadata__n657_3m_naExcluded.csv"

mb = pd.read_csv(file_path_mb_data, index_col=0)
mb = mb.T
metadata = pd.read_csv(file_path_metadata)

covars = ["StudyCenter___edmonton_ndnominal",
             "StudyCenter___vancouver_ndnominal",
             "StudyCenter___toronto_ndnominal",
             "StudyCenter___winnipeg_ndnominal"]
outcome_and_ID_vars = ["diseasestatus_3y_binary",
                       "SampleID_continuous"]

def at_most_one(row, col_names):
    ones = 0
    for col_name in col_names:
        if row[col_name] == 1:
            ones += 1
    return ones == 1

bad_rows = []
for idx, row in metadata.iterrows():
    if not at_most_one(row, covars):
        bad_rows += [row]
assert(len(bad_rows) == 0)

metadata = metadata[covars + outcome_and_ID_vars].dropna()

# Match the samples between the metadata and microbiome data.
metadata["SampleID"] = metadata["SampleID_continuous"].astype(str)
intersection = [ID for ID in metadata["SampleID"] if ID in mb.index]
metadata = metadata.loc[metadata["SampleID"].isin(intersection), ]
metadata = metadata.set_index("SampleID").sort_index()
mb = mb.loc[intersection, ].sort_index()
assert(all(metadata.index == mb.index))

# Define inputs for HSIC lasso.
OTU_IDs = mb.columns.values
X = np.array(mb)
Y = np.array(metadata[outcome_and_ID_vars[0]])
X_covars = np.array(metadata[covars])

"""
Setting B=5 performs vanilla HSIC lasso.
"""
d = mb.shape[1]-1
hsic_lasso = HSICLasso()
hsic_lasso.input(X, Y, featname=OTU_IDs)
hsic_lasso.classification(d, x_kernel="Jaccard", covars=X_covars, covars_kernel="Jaccard", B=5)
hsic_lasso.dump()
