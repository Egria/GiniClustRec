import argparse
import pandas
import numpy
import warnings
import os
from matplotlib import pyplot
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
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
        giniIndex = pandas.DataFrame({"gini":gini})
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
    means = exprM_raw_counts_filter.mean(axis=1)
    log2_maxs = numpy.log2(maxs + 0.1)
    exprM_stat1 = pandas.DataFrame({"Maxs": maxs, "Gini": giniIndex["gini"], "log2.Maxs": log2_maxs})
    exprM_stat1 = exprM_stat1[(exprM_stat1["log2.Maxs"] > expr_params.log2_expr_cutoffl) & (exprM_stat1["log2.Maxs"] <= expr_params.log2_expr_cutoffh)]
    log2_maxs = exprM_stat1["log2.Maxs"]
    gini = exprM_stat1["Gini"]
    maxs = exprM_stat1["Maxs"]
    # fitting in max-gini space
    x = exprM_stat1["log2.Maxs"].to_numpy()
    y = exprM_stat1["Gini"].to_numpy()
    fitted = lowess(endog=y, exog=x, frac=expr_params.span, it=0, return_sorted=False)
    residuals = y - fitted
    exprM_stat1 = pandas.DataFrame({
        "Maxs": exprM_stat1["Maxs"].to_numpy(),
        "Gini": y,
        "log2.Maxs": x,
        "Norm.Gini": residuals,
        "Gini.fitted": fitted
    }, index=exprM_stat1.index)

    # remove 25% of first round outlier genes, do second round loess
    gini_loess_fit_residual = pandas.Series(exprM_stat1["Norm.Gini"].to_numpy(), index=exprM_stat1.index)
    pos_res = gini_loess_fit_residual[gini_loess_fit_residual > 0]
    if len(pos_res) > 0:
        thresh_outlier = numpy.quantile(pos_res.to_numpy(), expr_params.outlier_remove)
    else:
        thresh_outlier = numpy.inf
    id_genes_loess_fit = (gini_loess_fit_residual < thresh_outlier).to_numpy()
    id_outliers_loess_fit = ~id_genes_loess_fit
    log2_maxs_genes = log2_maxs[id_genes_loess_fit]
    log2_maxs_outliers = log2_maxs[id_outliers_loess_fit]

    # second round loess
    x2 = log2_maxs_genes.to_numpy()
    y2 = exprM_stat1.loc[id_genes_loess_fit, "Gini"].to_numpy()
    if len(x2)>=2:
        fitted2 = lowess(endog=y2, exog=x2, frac=expr_params.span, it=0, return_sorted=False)
    else:
        fitted2 = numpy.full_like(y2, numpy.nan, dtype=float)
    gini_loess_fit_2_predict = pandas.Series(fitted2, index=log2_maxs_genes.index)

    # plot second round fit
    gini_loess_fit_2_x_y = pandas.DataFrame({"x": log2_maxs_genes.values, "y":gini_loess_fit_2_predict.values}, index = log2_maxs_genes.index)
    gini_loess_fit_2_x_y_uniq = gini_loess_fit_2_x_y.drop_duplicates()
    gini_loess_fit_2_x_y_uniq = gini_loess_fit_2_x_y_uniq.sort_values(by = "x")
    order = numpy.argsort(log2_maxs_genes.values)
    log2_maxs_genes_sorted = pandas.Series(log2_maxs_genes.values[order], index=log2_maxs_genes.index.values[order])
    gini_loess_fit_2_predict_sorted = pandas.Series(gini_loess_fit_2_predict.values[order], index=log2_maxs_genes.index.values[order])

    # using gini_loess_fit_2 as model, predict gini value for those outlier which are not used for build model
    # for each max in outliers set, find the id of max value which is most close in fitted data set
    loc_outliers = numpy.searchsorted(log2_maxs_genes_sorted, log2_maxs_outliers, side="left")
    loc_outliers = numpy.minimum(loc_outliers, len(log2_maxs_genes_sorted) - 1)
    outlier_max_in_fit = pandas.DataFrame({
        "log2_Maxs_outliers": log2_maxs_outliers,
        "loc_outliers": loc_outliers,
        "log2_Maxs_genes_at_loc": log2_maxs_genes_sorted.values[loc_outliers]
    })
    outliers = pandas.DataFrame({"id": numpy.arange(0, len(log2_maxs_outliers)), "value": log2_maxs_outliers})
    def second_round_fit(x: pandas.Series) -> float:
        id = int(x.iloc[0])
        value = x.iloc[1]
        if (value == log2_maxs_genes_sorted.iloc[loc_outliers[id]]):
            subset = gini_loess_fit_2_x_y_uniq.loc[gini_loess_fit_2_x_y_uniq["x"]>= value, "y"]
            subset = pandas.Series(subset.squeeze(), index=subset.index)
            if not subset.empty:
                return float(subset.iloc[0])
            else:
                return numpy.nan
        else:
            if loc_outliers[id] > 0:
                i = loc_outliers[id]
                x0 = log2_maxs_genes_sorted.iloc[i-1]
                x1 = log2_maxs_genes_sorted.iloc[i]
                y0 = gini_loess_fit_2_predict_sorted.iloc[i-1]
                y1 = gini_loess_fit_2_predict_sorted.iloc[i]
                return float(y0 + (y1 - y0) * (value - x0) / (x1 - x0))
            else:
                x0 = log2_maxs_genes_sorted.iloc[0]
                x1 = log2_maxs_genes_sorted.iloc[1]
                y0 = gini_loess_fit_2_predict_sorted.iloc[0]
                y1 = gini_loess_fit_2_predict_sorted.iloc[1]
                return float(y1 - (y1 - y0) * (x1 - value) / (x1 - x0))
    gini_outliers_predict = outliers.apply(second_round_fit, axis=1)

    # plot outliers predict results
    outliers_predict_x_y_uniq = pandas.DataFrame({
        "x": log2_maxs_outliers,
        "y": gini_outliers_predict
    }).drop_duplicates()
    #outliers_predict_x_y_uniq.plot(x="x", y="y", kind="scatter")
    #pyplot.xlabel("log2(Max expression outliers)")
    #pyplot.ylabel("Predicted Gini")
    #pyplot.title("Outlier Predictions")
    #pyplot.show()

    # plot whole fit 2
    gini_loess_fit_2_full_x_y_uniq = pandas.concat([gini_loess_fit_2_x_y_uniq, outliers_predict_x_y_uniq], axis=0)
    #gini_loess_fit_2_full_x_y_uniq.plot(x="x", y="y", kind="scatter")
    #pyplot.xlabel("log2(Max expression outliers)")
    #pyplot.ylabel("Fitted/Predicted Gini")
    #pyplot.title("LOESS fit (inliers) + predicted outliers")
    #pyplot.show()

    # calculate normalized_gini_score2
    normalized_gini_score2 = numpy.zeros(len(gini_loess_fit_residual), dtype=float)
    residuals2 = y2 - fitted2
    normalized_gini_score2[id_genes_loess_fit] = residuals2
    normalized_gini_score2[id_outliers_loess_fit] = gini[id_outliers_loess_fit].to_numpy() - gini_outliers_predict
    gini_fitted2 = gini - normalized_gini_score2
    print(gini_fitted2)
    print(exprM_stat1.shape)
    exprM_stat1 = exprM_stat1.loc[:, ["Maxs", "Gini", "log2.Maxs", "Gini.fitted", "Norm.Gini"]].copy()
    exprM_stat1["Gini.fitted2"] = gini_fitted2
    exprM_stat1["Norm.Gini2"] = normalized_gini_score2
    z = (exprM_stat1["Norm.Gini2"] - exprM_stat1["Norm.Gini2"].mean()) / exprM_stat1["Norm.Gini2"].std(ddof=1)
    gini_pvalue = stats.norm.cdf(-numpy.abs(z))
    exprM_stat2 = exprM_stat1.copy()
    exprM_stat2["Gini.pvalue"] = gini_pvalue

    # for each measurement, first ranked by themselves.
    # identify High Gini Genes with Norm.Gini
    exprM_stat2 = exprM_stat2.sort_values("Norm.Gini2", ascending=False)
    genelist_highNormGini = exprM_stat2.loc[exprM_stat2["Norm.Gini2"] > expr_params.norm_gini_cutoff].index
    print(len(genelist_highNormGini))

    # identify High Gini Genes with pvalue
    exprM_stat2 = exprM_stat2.sort_values("Gini.pvalue", ascending=True)
    genelist_top_pvalue = exprM_stat2.loc[(exprM_stat2["Gini.pvalue"] < expr_params.gini_pvalue_cutoff)&(exprM_stat2["Norm.Gini2"] > 0)].index
    print(len(genelist_top_pvalue))

    # plot figures
    xall = exprM_stat2["log2.Maxs"]
    yall = exprM_stat2["Gini"]
    yfit2 = gini_fitted2

    # histogram of pvalue
    pvals = exprM_stat2["Gini.pvalue"]
    neglog10_p = -numpy.log10(pvals)
    main_title = (f"Histogram of -log10(Gini.pvalue)\n"
                  f"cutoff={expr_params.gini_pvalue_cutoff}\n"
                  f"Gene num = {len(genelist_top_pvalue)}")
    out_path = os.path.join(main_args.out, "figures", f"{expr_params.experiment_id}_histogram_of_Normalized.Gini.Score.pdf")
    pyplot.figure(figsize=(6,6))
    pyplot.hist(neglog10_p, bins=100, color="lightgray", edgecolor="black")
    pyplot.axvline(-numpy.log10(expr_params.gini_pvalue_cutoff), color="red", linestyle="-")
    pyplot.title(main_title)
    pyplot.xlabel("-log10(Gini.pvalue)")
    pyplot.ylabel("GeneCount")
    pyplot.tight_layout()
    pyplot.savefig(out_path)







