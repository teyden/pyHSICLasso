import sys

import pandas as pd
import numpy as np

from collections import namedtuple

import pprint
from pyHSICLasso import HSICLasso, kernel_tools

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score

KernelMethods = namedtuple("KernelMethods", ["y", "x", "covars"])

CASES_OLD = {
    "diseasestatus_3y_binary": {
        "3m": {
            "fp_microbiome_data": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__4__3m-ds3y_microbiome.csv",
            "fp_metadata": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__4__3m-ds3y_metadata.csv",
            "timepoint": "3m",
            "outcome": "diseasestatus_3y_binary"
        },
        "1y": {
            "fp_microbiome_data": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__6__1y-ds3y_microbiome.csv",
            "fp_metadata": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__6__1y-ds3y_metadata.csv",
            "timepoint": "1y",
            "outcome": "diseasestatus_3y_binary"
        }
    },
    "diseasestatus_5y_binary": {
        "3m": {
            "fp_microbiome_data": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__5__3m-ds5y_microbiome.csv",
            "fp_metadata": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__5__3m-ds5y_metadata.csv",
            "timepoint": "3m",
            "outcome": "diseasestatus_5y_binary"
        },
        "1y": {
            "fp_microbiome_data": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__7__1y-ds5y_microbiome.csv",
            "fp_metadata": "/Users/teyden/Desktop/splitted-data-from-server/data_ID__7__1y-ds5y_metadata.csv",
            "timepoint": "1y",
            "outcome": "diseasestatus_5y_binary"
        }
    }
}

CASES_OLD_2 = {
    "diseasestatus_3y_binary": {
        "3m_1": {
            "fp_microbiome_data": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__9__3m-ds3y_microbiome__train.csv",
            "fp_metadata": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__9__3m-ds3y_metadata__train.csv",
            "fp_microbiome_data_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__9__3m-ds3y_microbiome__test.csv",
            "fp_metadata_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__9__3m-ds3y_metadata__test.csv",
            "timepoint": "3m",
            "outcome": "diseasestatus_3y_binary"
        },
        "3m_2": {
            "fp_microbiome_data": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__17__3m-ds3y_microbiome__train.csv",
            "fp_metadata": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__17__3m-ds3y_metadata__train.csv",
            "fp_microbiome_data_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__17__3m-ds3y_microbiome__test.csv",
            "fp_metadata_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__17__3m-ds3y_metadata__test.csv",
            "timepoint": "3m",
            "outcome": "diseasestatus_3y_binary"
        },
        "1y_1": {
            "fp_microbiome_data": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__15__1y-ds3y_microbiome__train.csv",
            "fp_metadata": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__15__1y-ds3y_metadata__train.csv",
            "fp_microbiome_data_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__15__1y-ds3y_microbiome__test.csv",
            "fp_metadata_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__15__1y-ds3y_metadata__test.csv",
            "timepoint": "1y",
            "outcome": "diseasestatus_3y_binary"
        },
        "1y_2": {
            "fp_microbiome_data": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__11__1y-ds3y_microbiome__train.csv",
            "fp_metadata": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__11__1y-ds3y_metadata__train.csv",
            "fp_microbiome_data_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__11__1y-ds3y_microbiome__test.csv",
            "fp_metadata_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__11__1y-ds3y_metadata__test.csv",
            "timepoint": "1y",
            "outcome": "diseasestatus_3y_binary"
        }
    },
    "diseasestatus_5y_binary": {
        "3m": {
            "fp_microbiome_data": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__18__3m-ds5y_microbiome__train.csv",
            "fp_metadata": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__18__3m-ds5y_metadata__train.csv",
            "fp_microbiome_data_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__18__3m-ds5y_microbiome__test.csv",
            "fp_metadata_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__18__3m-ds5y_metadata__test.csv",
            "timepoint": "3m",
            "outcome": "diseasestatus_5y_binary"
        },
        "1y": {
            "fp_microbiome_data": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__20__1y-ds5y_microbiome__train.csv",
            "fp_metadata": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__20__1y-ds5y_metadata__train.csv",
            "fp_microbiome_data_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__20__1y-ds5y_microbiome__test.csv",
            "fp_metadata_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__20__1y-ds5y_metadata__test.csv",
            "timepoint": "1y",
            "outcome": "diseasestatus_5y_binary"
        }
    }
}

CASES = {
    "diseasestatus_3y_binary": {
        "3m": {
            "fp_microbiome_data": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__9__3m-ds3y_microbiome__train.csv",
            "fp_metadata": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__9__3m-ds3y_metadata__train.csv",
            "fp_microbiome_data_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__9__3m-ds3y_microbiome__test.csv",
            "fp_metadata_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__9__3m-ds3y_metadata__test.csv",
            "timepoint": "3m",
            "outcome": "diseasestatus_3y_binary"
        },
        "1y": {
            "fp_microbiome_data": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__15__1y-ds3y_microbiome__train.csv",
            "fp_metadata": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__15__1y-ds3y_metadata__train.csv",
            "fp_microbiome_data_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__15__1y-ds3y_microbiome__test.csv",
            "fp_metadata_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__15__1y-ds3y_metadata__test.csv",
            "timepoint": "1y",
            "outcome": "diseasestatus_3y_binary"
        }
    },
    "diseasestatus_5y_binary": {
        "3m": {
            "fp_microbiome_data": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__18__3m-ds5y_microbiome__train.csv",
            "fp_metadata": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__18__3m-ds5y_metadata__train.csv",
            "fp_microbiome_data_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__18__3m-ds5y_microbiome__test.csv",
            "fp_metadata_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__18__3m-ds5y_metadata__test.csv",
            "timepoint": "3m",
            "outcome": "diseasestatus_5y_binary"
        },
        "1y": {
            "fp_microbiome_data": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__20__1y-ds5y_microbiome__train.csv",
            "fp_metadata": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__20__1y-ds5y_metadata__train.csv",
            "fp_microbiome_data_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__20__1y-ds5y_microbiome__test.csv",
            "fp_metadata_test": "~/Projects/asthma/data-objects/cluster-data/data-for-testing/data_ID__20__1y-ds5y_metadata__test.csv",
            "timepoint": "1y",
            "outcome": "diseasestatus_5y_binary"
        }
    }
}

KERNELS = ["Jaccard", "BrayCurtis", "Gaussian"]

def load_data(file_path_mb_data, file_path_metadata, outcome, timepoint, add_constant=False, add_pseudo_species=False):
    # Read in the data
    mb = pd.read_csv(file_path_mb_data, index_col=0)
    metadata = pd.read_csv(file_path_metadata, index_col=0)

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
    OTU_IDs = list(mb.columns.values)
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
        OTU_IDs.append("OTU_pseudo")

    X_df = pd.DataFrame(X, columns=OTU_IDs) + pd.DataFrame(X_covars, columns=covars)

    return X, Y, X_covars, OTU_IDs, covars

def run_hsic_lasso(file_path_mb_data, file_path_metadata, outcome, timepoint, kernel_methods, add_constant=False, add_pseudo_species=False):
    # if add_pseudo_species and add_constant:
    #     raise ValueError("Cannot add constant and add a pseudo species at the same time. Pick one.")

    print("\n")
    print("#"*100)
    print("######## Testing microbiome data timepoint {} for outcome {}".format(timepoint, outcome))
    print("#"*100)

    print(kernel_methods)

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
    # elif add_pseudo_species:
    #     min = kernel_tools.find_nonzero_min(X)
    #     X = kernel_tools.add_pseudo_species(X, min)
    #     d = d + 1

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
                              zero_adjust=False)
    hsic_lasso.dump()

    return hsic_lasso

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

def generate_hsic_results(RESULT):
    selection_results_hsic = {}
    for outcome, timepoint_cases in CASES.items():
        selection_results_hsic[outcome] = {}
        for timepoint, case in timepoint_cases.items():
            case_result = {}
            kernel_results = {}
            case_result["kernels"] = kernel_results
            for i, kernel in enumerate(KERNELS):
                kernel_methods = KernelMethods(y="Delta", x=kernel, covars=kernel)
                hsic_lasso = run_hsic_lasso(case["fp_microbiome_data"],
                                            case["fp_metadata"],
                                            case["outcome"],
                                            case["timepoint"],
                                            kernel_methods,
                                            add_constant=True,
                                            add_pseudo_species=True)

                kernel_results[kernel] = {
                    # "hsic_lasso": hsic_lasso,
                    "features": hsic_lasso.get_features()
                }

                if i == 0:
                    intersection = set(hsic_lasso.get_features())
                else:
                    intersection = case_result["overlapping_features"] & set(hsic_lasso.get_features())

                case_result["overlapping_features"] = intersection
            selection_results_hsic[outcome][timepoint] = case_result
    pprint.pprint(selection_results_hsic)
    RESULT["hsic_lasso"] = selection_results_hsic
    return RESULT

def generate_lr_rfe_results(RESULT):
    selection_results = {}
    for outcome, timepoint_cases in CASES.items():
        selection_results[outcome] = {}
        for timepoint, case in timepoint_cases.items():
            X, Y, X_covars, OTU_IDs, covars = load_data(case["fp_microbiome_data"],
                                        case["fp_metadata"],
                                        case["outcome"],
                                        case["timepoint"],
                                        add_constant=False,
                                        add_pseudo_species=True)

            X = pd.DataFrame(X, columns=OTU_IDs)

            # X_covars = pd.DataFrame(X_covars, columns=covars)
            # X = pd.concat([X, X_covars.reindex(X.index)], axis=1)
            # X = X + X_covars
            # print(X)
            # X = pd.DataFrame(X, columns=OTU_IDs)
            # K = kernel_tools.kernel_custom(X, kernel=kernel, zero_adjust=False)

            estimator = LogisticRegression(max_iter=10)
            selector = RFECV(estimator, step=1, cv=5)
            selector = selector.fit(X, Y)
            print(selector.n_features_)

            selected_features = X.columns.values[selector.support_]

            case_result = {
                "features": selected_features,
                "n_features": selector.n_features_,
                # "selector": selector
            }

            selection_results[outcome][timepoint] = case_result
    pprint.pprint(selection_results)
    RESULT["lr_rfecv"] = selection_results
    return RESULT

def generate_lr_lasso_results(RESULT):
    selection_results = {}
    for outcome, timepoint_cases in CASES.items():
        selection_results[outcome] = {}
        for timepoint, case in timepoint_cases.items():
            # This represents training data only
            X, Y, X_covars, OTU_IDs, covars = load_data(case["fp_microbiome_data"],
                                        case["fp_metadata"],
                                        case["outcome"],
                                        case["timepoint"],
                                        add_constant=False,
                                        add_pseudo_species=True)

            X = pd.DataFrame(X, columns=OTU_IDs)

            estimator = LogisticRegression(penalty="l1", C=1.0, solver="liblinear")
            estimator.fit(X, Y)

            selected_features = X.columns.values[estimator.coef_[0] != 0]

            case_result = {
                "features": selected_features,
                # "estimator": estimator
            }
            selection_results[outcome][timepoint] = case_result

    RESULT["lr_lasso"] = selection_results
    return RESULT

HSIC_RESULT = None
RFE_RESULT = None

if __name__ == "__main__":
    args = sys.argv
    method1 = ""
    method2 = ""
    method3 = ""
    if len(args) > 1:
        method1 = args[1]
    if len(args) > 2:
        method2 = args[2]
    if len(args) > 3:
        method3 = args[3]

    method1 = "--hsiclasso"
    
    methods = [method1, method2, method3]

    RESULT = {}
    if "--hsiclasso" in methods:
        RESULT = generate_hsic_results(RESULT)

    if "--lrrfe" in methods:
        RESULT = generate_lr_rfe_results(RESULT)

    if "--lrlasso" in methods:
        RESULT = generate_lr_lasso_results(RESULT)

    pprint.pprint(RESULT)

    import pickle
    f = open("/Users/teyden/Desktop/feature_selection_results.pkl", "wb")
    pickle.dump(RESULT, f)
    f.close()

    # print("\n")
    # print("*"*100)
    # print("\n")
    # overlapping = {}
    # i = 0
    # for method, result in RESULT.items():
    #     outcomes = result.keys()
    #     timepoints = result[outcomes[0]].keys()
    #     for outcome in outcomes:
    #         overlapping[outcome] = {}
    #         for timepoint in timepoints:
    #             if i == 0:
    #                 intersection = set(result[outcome][timepoint])
    #             else:
    #                 intersection = overlapping[outcome][timepoint] & set(result[outcome][timepoint])
    #             overlapping[outcome][timepoint] = intersection
    #
    # pprint.pprint(overlapping)