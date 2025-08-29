import argparse
import pandas
import numpy
import warnings
import os
from .GiniClust_parameters import GiniClust_Parameters

def calcul_gini(x: pandas.Series, unbiased: bool = True, na_rm:bool = False) -> float:
    if not pandas.api.types.is_numeric_dtype(x):
        warnings.warn("'x' is not numeric; returning NA")
        return numpy.nan
    if (not na_rm) and (x.isna().any()):
        raise ValueError("'x' contains NaNs")
    if na_rm:
        x = x.dropna()
    n = len(x)
    mu = x.mean()
    if unbiased:
        N = n * (n - 1)
    else:
        N = n * n
    ox = numpy.sort(x.to_numpy())
    weights = 2 * numpy.arange(1, n + 1) - n - 1
    dsum = numpy.dot(ox, weights)
    return dsum / (mu * N)

def giniClust_fitting(exprM_raw_counts_filter:pandas.DataFrame, main_args: argparse.Namespace, expr_params: GiniClust_Parameters) -> pandas.DataFrame:
    assert main_args.type in ['RNA-seq', 'qPCR']
    if main_args.type == 'RNA-seq':
        gini = exprM_raw_counts_filter.apply(lambda x: calcul_gini(x.astype(float)), axis=1)
        n_rows = exprM_raw_counts_filter.shape[0]
        giniIndex = pandas.DataFrame({"Row": range(1, n_rows + 1), "Gini": gini.values if hasattr(gini, "values") else gini})
    elif main_args.type == 'qPCR':
        giniIndex1 = pandas.DataFrame(exprM_raw_counts_filter.apply(lambda x: calcul_gini(x.astype(float)), axis=1),columns=["gini1"])
        giniIndex2 = pandas.DataFrame(exprM_raw_counts_filter.add(0.00001).apply(lambda x: calcul_gini(1.0 / x.astype(float)),axis=1),columns=["gini2"])
        giniIndex = pandas.concat([giniIndex1, giniIndex2], axis=1)
        giniIndex["gini2_sign"] = -giniIndex["gini2"]
        giniIndex["gini"] = giniIndex.max(axis=1)
        giniIndex = giniIndex.dropna()
        # significant difference from R code
        giniIndex["gini_sign"] = numpy.where(giniIndex.iloc[:, 0] > giniIndex.iloc[:, 1], "up-regulation", "down-regulation")
        giniIndex.to_csv(os.path.join(main_args.out,f"{expr_params.experiment_id}_bi-directional.GiniIndexTable.csv"), sep=',', index=True, header=True, quoting=0)
    else:
        raise ValueError(f"Unknown type '{main_args.type}'")

    maxs = exprM_raw_counts_filter.max(axis=1)

