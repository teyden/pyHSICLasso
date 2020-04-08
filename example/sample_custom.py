import pandas as pd
import numpy as np

from collections import namedtuple

from pyHSICLasso import HSICLasso, kernel_tools

KernelMethods = namedtuple("KernelMethods", ["y", "x", "covars"])

def run_hsic_lasso(file_path_mb_data, file_path_metadata, outcome, timepoint, kernel_methods, add_constant=False, add_pseudo_species=False):
    if add_pseudo_species and add_constant:
        raise ValueError("Cannot add constant and add a pseudo species at the same time. Pick one.")

    print("\n")
    print("#"*100)
    print("######## Testing microbiome data timepoint {} for outcome {}".format(timepoint, outcome))
    print("#"*100)

    # Read in the data
    mb = pd.read_csv(file_path_mb_data, index_col=0)
    metadata = pd.read_csv(file_path_metadata, index_col=0)

    print(mb.shape)
    print(metadata.shape)

    covars = ["StudyCenter___edmonton_ndnominal",
              "StudyCenter___vancouver_ndnominal",
              "StudyCenter___toronto_ndnominal",
              "StudyCenter___winnipeg_ndnominal",
              "AgeAtVisit_continuous",
              "batch___1_ndnominal",
              "batch___2_ndnominal",
              "batch___3_ndnominal",
              "batch___4_ndnominal",
              "batch___5_ndnominal"]
    outcome_and_ID_vars = [outcome,
                           "SampleID_continuous"]

    metadata = metadata[covars + outcome_and_ID_vars].dropna()

    # Match the samples between the metadata and microbiome data.
    # metadata["SampleID"] = metadata["SampleID_continuous"].astype(str)
    intersection = [ID for ID in metadata["SampleID_continuous"] if ID in mb.index]
    metadata = metadata.loc[metadata["SampleID_continuous"].isin(intersection),]
    metadata = metadata.set_index("SampleID_continuous").sort_index()
    mb = mb.loc[intersection,].sort_index()
    assert(all(metadata.index == mb.index))

    # Define inputs for HSIC lasso.
    OTU_IDs = mb.columns.values
    X = np.array(mb)
    Y = np.array(metadata[outcome_and_ID_vars[0]])
    X_covars = np.array(metadata[covars])
    d = mb.shape[1] - 1

    if add_constant:
        X = X + 1
    elif add_pseudo_species:
        min = kernel_tools.find_nonzero_min(X)
        X = kernel_tools.add_pseudo_species(X, min)
        d = d + 1

    """
    Setting B=5 performs vanilla HSIC lasso.
    """
    hsic_lasso = HSICLasso()
    hsic_lasso.input(X, Y, featname=OTU_IDs)
    hsic_lasso.classification(d,
                              y_kernel=kernel_methods.y,
                              x_kernel=kernel_methods.x,
                              covars_kernel=kernel_methods.covars,
                              covars=X_covars,
                              B=5,
                              zero_adjust=add_pseudo_species)
    hsic_lasso.dump()

    return hsic_lasso.get_features()

"""
Notes: adding a constant of 1 to the microbiome counts prevented the Jaccard distances from producing nan's
"""
# | Order | Feature      | Score | Top-5 Related Feature (Relatedness Score)                                          |
# | 1     | OTU_644      | 1.000 | OTU_102      (0.097), OTU_31       (0.097), OTU_314      (0.094), OTU_227      (0.088), OTU_46       (0.086)|
# | 2     | OTU_93       | 0.994 | OTU_87       (0.182), OTU_199      (0.174), OTU_496      (0.160), OTU_83       (0.156), OTU_969      (0.144)|
# | 3     | OTU_968      | 0.829 | OTU_83       (0.124), OTU_448      (0.115), OTU_80       (0.115), OTU_189      (0.114), OTU_87       (0.108)|
# | 4     | OTU_136      | 0.746 | OTU_97       (0.125), OTU_961      (0.124), OTU_385      (0.111), OTU_969      (0.108), OTU_688      (0.101)|
# | 5     | OTU_380      | 0.741 | OTU_49       (0.128), OTU_7        (0.086), OTU_94       (0.078), OTU_109      (0.077), OTU_20       (0.076)|
# | 6     | OTU_20       | 0.733 | OTU_49       (0.187), OTU_48       (0.155), OTU_4        (0.151), OTU_24       (0.149), OTU_2        (0.138)|
# | 7     | OTU_612      | 0.699 | OTU_385      (0.192), OTU_199      (0.184), OTU_603      (0.182), OTU_83       (0.181), OTU_436      (0.179)|
# | 8     | OTU_795      | 0.609 | OTU_75       (0.165), OTU_229      (0.138), OTU_232      (0.136), OTU_43       (0.136), OTU_227      (0.135)|
# | 9     | OTU_4        | 0.607 | OTU_31       (0.218), OTU_49       (0.216), OTU_18       (0.200), OTU_98       (0.188), OTU_94       (0.184)|

if __name__ == "__main__":
    # file_path_mb_data = "/Users/teyden/Projects/asthma/data/child-study-data-jan-2019/processed/microbiome-data/filtered_ra.3m.csv"
    # file_path_metadata = "~/Projects/asthma/data/child-study-data-jan-2019/processed/metadata/merged-metadata/merged_metadata__n657_3m_naExcluded.csv"
    # run_hsic_lasso(file_path_mb_data, file_path_metadata, "diseasestatus_3y_binary", "3m")

    store = {
        1: {
            "fp_microbiome_data": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__4__3m-ds3y_microbiome.csv",
            "fp_metadata": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__4__3m-ds3y_metadata.csv",
            "timepoint": "3m",
            "outcome": "diseasestatus_3y_binary"
        },
        2: {
            "fp_microbiome_data": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__6__1y-ds3y_microbiome.csv",
            "fp_metadata": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__6__1y-ds3y_metadata.csv",
            "timepoint": "1y",
            "outcome": "diseasestatus_3y_binary"
        },
        3: {
            "fp_microbiome_data": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__5__3m-ds5y_microbiome.csv",
            "fp_metadata": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__5__3m-ds5y_metadata.csv",
            "timepoint": "3m",
            "outcome": "diseasestatus_5y_binary"
        },
        4: {
            "fp_microbiome_data": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__7__1y-ds5y_microbiome.csv",
            "fp_metadata": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__7__1y-ds5y_metadata.csv",
            "timepoint": "1y",
            "outcome": "diseasestatus_5y_binary"
        },
    }

    for key, case in store.items():
        kernel_methods = KernelMethods(y="Delta", x="BrayCurtis", covars="BrayCurtis")
        print(kernel_methods)
        features = run_hsic_lasso(case["fp_microbiome_data"],
                                  case["fp_metadata"],
                                  case["outcome"],
                                  case["timepoint"],
                                  kernel_methods,
                                  add_pseudo_species=True)
        print(features)

    # ## Which OTUs intersect in predicting 3 year asthma between the 3month and 1year samples?
    # intersection_ds3y = [otu for otu in features_3m_ds3y if otu in features_1y_ds3y]
    #
    # ## Which one differ?
    # uniq_3m_ds3y = [otu for otu in features_3m_ds3y if otu not in features_1y_ds3y]
    # uniq_1y_ds3y = [otu for otu in features_1y_ds3y if otu not in features_3m_ds3y]
    #
    # print("IN COMMON BETWEEN BOTH TIME POINTS FOR DS3Y")
    # print(intersection_ds3y)
    #
    # ## Which OTUs intersect in predicting 3 year asthma between the 3month and 1year samples?
    # intersection_ds5y = [otu for otu in features_3m_ds5y if otu in features_1y_ds5y]
    #
    # print("IN COMMON BETWEEN BOTH TIME POINTS FOR DS5Y")
    # print(intersection_ds5y)
