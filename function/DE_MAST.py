import argparse
import numpy
import pandas
import os
from sklearn import metrics
from .GiniClust_parameters import GiniClust_Parameters
from .MAST.ZLM_result import zlm_single_cell_assay


def mean_in_log2space(x, pseudo_count=1.0):
    return numpy.log2((numpy.power(2.0, x)-pseudo_count).mean() + pseudo_count)

def stat_log2(data_m: pandas.DataFrame, group_v: pandas.Series, pseudo_count=1.0) -> pandas.DataFrame:
    group_v = group_v.reindex(data_m.columns)
    levels = pandas.Index(pandas.unique(group_v))
    if set(levels) == {0, 1}:
        g0, g1 = 0, 1
    else:
        g0, g1 = sorted(levels.tolist())[:2]
    cols0 = data_m.columns[group_v == g0]
    cols1 = data_m.columns[group_v == g1]
    mean0 = data_m.loc[:, cols0].apply(mean_in_log2space, axis=1, pseudo_count=pseudo_count)
    mean1 = data_m.loc[:, cols1].apply(mean_in_log2space, axis=1, pseudo_count=pseudo_count)
    df = pandas.DataFrame({"log2.mean.group0":mean0, "log2.mean.group1":mean1}, index=data_m.index)
    df["log2_fc"] = df["log2.mean.group1"] - df["log2.mean.group0"]
    return df

def v_auc(data_v, group_v) -> float:
    scores = data_v.to_numpy(dtype=float)
    labels = group_v.reindex(data_v.index).to_numpy()
    labels = pandas.Series(labels).astype("category")
    labels = labels.cat.codes.to_numpy()
    return float(metrics.roc_auc_score(labels,scores))

def m_auc(data_m:pandas.DataFrame, group_v:pandas.Series) -> pandas.Series:
    group_v = group_v.reindex(data_m.columns)
    aucs = data_m.apply(lambda row: v_auc(row, group_v), axis=1)
    return aucs

def de_mast(exprM_rawCounts_filter: pandas.DataFrame, rare_cells_list_all:dict, c_membership: pandas.Series, clustering_membership_r: pandas.DataFrame, main_args:argparse.Namespace, expr_params:GiniClust_Parameters):
    pseudo_count = 0.1
    data_used_log2 = pandas.DataFrame(numpy.log2(exprM_rawCounts_filter + pseudo_count))
    cells_symbol_list1 = clustering_membership_r.loc[clustering_membership_r["cluster.ID"]=="Cluster_1", "cell.ID"].tolist()
    cells_coord_list1 = [data_used_log2.columns.get_loc(c) for c in cells_symbol_list1 if c in data_used_log2.columns]
    c_membership = c_membership.reindex(data_used_log2.columns)
    for rare_cluster, cells_symbol_list2 in rare_cells_list_all.items():
        cells_coord_list2 = [data_used_log2.columns.get_loc(c) for c in cells_symbol_list2 if c in data_used_log2.columns]
        data_used_log2_ordered = pandas.concat([data_used_log2.iloc[:,cells_coord_list1], data_used_log2.iloc[:, cells_coord_list2]], axis=1)
        group_v = pandas.Series([0]*len(cells_coord_list1) + [1]*len(cells_coord_list2), index=data_used_log2_ordered.columns)
        # output
        log2_stat_result = stat_log2(data_used_log2_ordered, group_v, pseudo_count)
        auc = m_auc(data_used_log2_ordered, group_v).rename("Auc")
        bigtable = log2_stat_result.join(auc)

        diff_cutoff = 1.0
        de = bigtable[bigtable["log2_fc"] > diff_cutoff]
        print(de.shape)
        data_1 = data_used_log2.iloc[:, cells_coord_list1]
        data_2 = data_used_log2.iloc[:, cells_coord_list2]
        genes_list = de.index.tolist()
        log2fold_change = pandas.DataFrame({"gene.name": genes_list, "log2_foldchange": de["log2_fc"].values}).set_index("gene.name")
        counts = pandas.concat([data_1.loc[genes_list,:], data_2.loc[genes_list,:]], axis=1)
        groups = (["Cluster_1"] * len(cells_coord_list1) + [rare_cluster] * len(cells_coord_list2))
        groups = [str(g) for g in groups]
        data_for_mist = counts.reset_index().melt(id_vars="index", var_name="Subject.ID", value_name="Et").rename(columns={"index":"Gene"})
        col_to_group = dict(zip(counts.columns, groups))
        data_for_mist["Population"] = data_for_mist["Subject.ID"].map(col_to_group)
        data_for_mist["Number.of.Cells"] = 1
        data_for_mist = data_for_mist[["Gene", "Subject.ID", "Et", "Population", "Number.of.Cells"]]
        vbeta = data_for_mist.copy()
        vbeta_1 = vbeta.copy()

        # MAST
        col_data = pandas.DataFrame({"Population": pandas.Categorical(groups, categories=["Cluster_1", rare_cluster])}, index=counts.columns)
        res = zlm_single_cell_assay(data_used_log2_ordered, group_v, pseudo_count, None, True, "bayesglm", True)
        print(f"Genes: {res.table.shape[0]}  Cells: {data_used_log2_ordered.shape[1]}")
        print("Per-cell metadata columns:", list(col_data.columns))
        print("\nMAST-like summary (first rows):")
        print(res.summary(logFC=True).datatable.head())

        coef_and_ci = res.summary(logFC=True).datatable.copy()
        #coef_and_ci = coef_and_ci[coef_and_ci["contrast"] != "(Intercept)"]
        #coef_and_ci["contrast"] = coef_and_ci["contrast"].str.replace("Population","Pop")
        print(coef_and_ci.head())

        lrt = res.lrTest("Population")
        zlm_lr_pvalue = (lrt.loc[lrt["test.type"] == "hurdle", ["primerid", "Pr(>Chisq)"]].rename(columns={"Pr(>Chisq)":"p_value"}).reset_index(drop=True))
        zlm_lr_pvalue = zlm_lr_pvalue.set_index("primerid")
        print(zlm_lr_pvalue.head())

        lrTest_table = zlm_lr_pvalue.merge(de,left_index=True,right_index=True,how="inner").reset_index()
        lrTest_table.columns = ["Gene", "p_value",f"log2.mean.Cluster_1", f"log2.mean.{rare_cluster}","log2fold_change", "Auc"]
        cluster_lrTest_table = lrTest_table.sort_values("Auc", ascending=False)
        print(cluster_lrTest_table.head())

        cluster_lrTest_table.to_csv(os.path.join(main_args.out,rare_cluster+".diff.gene.lrTest.results.csv"))

