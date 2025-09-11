import argparse
import matplotlib_venn
import numpy
import pandas
import warnings
import os
from matplotlib import pyplot
from sklearn import metrics
from scipy import stats
from .GiniClust_parameters import GiniClust_Parameters
from .MAST.ZLM_result import zlm_single_cell_assay
from .MAST.mast_like import mast_like


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

def de_mast(exprM_rawCounts_filter: pandas.DataFrame, rare_cells_list_all:dict, c_membership: pandas.Series, genelist_final: pandas.Index, clustering_membership_r: pandas.DataFrame, main_args:argparse.Namespace, expr_params:GiniClust_Parameters):
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
        warnings.filterwarnings("ignore")
        py_mast = False
        if py_mast:
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
            cluster_lrTest_table.set_index("Gene", inplace=True)

        else:
            mast_res = mast_like(
                data_log2=data_used_log2_ordered,
                group=group_v,
                pseudo_count=pseudo_count,
                covariates=None,  # plug a DataFrame of cell covariates here if you have (batch, nUMI, etc.)
                add_cdr=True
            )
            cluster_lrTest_table = mast_res.rename(columns={
                "log2.mean.group0": "log2.mean.Cluster_1",
                "log2.mean.group1": f"log2.mean.{rare_cluster}",
                "log2_fc": "log2fold_change",
                "p_comb": "p_value"  # use combined p as main test p-value
            }).loc[:, [f"log2.mean.{rare_cluster}", "log2.mean.Cluster_1", "log2fold_change", "Auc", "p_value", "q_comb"]]

            # sort like you did (you sorted by AUC in R just before saving)
            cluster_lrTest_table = cluster_lrTest_table.sort_values("Auc", ascending=False)



        print(cluster_lrTest_table.head())

        cluster_lrTest_table.to_csv(os.path.join(main_args.out,rare_cluster+".diff.gene.lrTest.results.csv"))

        # overlap fisher.test

        cluster_diffgene = cluster_lrTest_table.index[(cluster_lrTest_table["p_value"]<expr_params.lr_p_value_cutoff)&(numpy.abs(cluster_lrTest_table["log2fold_change"]) > expr_params.diff_cutoff)]
        print(len(cluster_diffgene))
        overlap_genes = set(cluster_diffgene).intersection(genelist_final)
        print(len(overlap_genes))
        area1 = len(cluster_diffgene)
        area2 = len(genelist_final)
        n12 = len(overlap_genes)
        cluster_overlap = numpy.array(
            [[n12, area1 - n12], [area2 - n12, exprM_rawCounts_filter.shape[0] - (area1 + area2 - n12)]], dtype=int)
        print(cluster_overlap)
        oddsratio, ft_pvalue = stats.fisher_exact(cluster_overlap, alternative="two-sided")

        # Venn diagram
        labels = (f"diffgene\n{area1}", f"HighGiniGenes\n{area2}\n{ft_pvalue:.2e}")
        pyplot.figure(figsize=(6, 6))
        matplotlib_venn.venn2(subsets=(area1 - n12, area2 - n12, n12), set_labels=labels, set_colors=("blue", "yellow"),
                              alpha=0.5)
        pyplot.savefig(
            os.path.join(main_args.out, "figures", f"{expr_params.experiment_id}_{rare_cluster}_diff_gene_overlap.pdf"))
        pyplot.close()

        # barplot visilization of gene expression for individual cells in orginal order
        overlap_genes = (set(cluster_lrTest_table.loc[cluster_lrTest_table["Auc"] > 0.98, ].index).intersection(set(genelist_final)))
        mycols = ["#D9D9D9"] + list(pyplot.cm.rainbow(numpy.linspace(0, 1, len(set(c_membership)) - 1)))
        if len(overlap_genes) > 0:
            for genei in overlap_genes:
                x1 = exprM_rawCounts_filter.loc[genei, cells_symbol_list1].astype(float).values
                x2 = exprM_rawCounts_filter.loc[genei, cells_symbol_list2].astype(float).values
                ylim = max(numpy.max(x1), numpy.max(x2)) * 1.2
                out_path = os.path.join(
                    main_args.out,
                    "figures",
                    f"{expr_params.experiment_id}_{rare_cluster}_overlapgene_rawCounts_bar_plot.{genei}.pdf"
                )
                fig, axes = pyplot.subplots(1, 2, figsize=(12, 6))
                axes[0].bar(numpy.arange(len(x2)), x2, color=mycols[2])
                axes[0].set_title(f"{genei} {rare_cluster}")
                axes[0].set_ylim([0, ylim])
                axes[0].set_ylabel("Counts/cell")
                axes[1].bar(numpy.arange(len(x1)), x1, color=mycols[1])
                axes[1].set_title("Cluster_1")
                axes[1].set_ylim([0, ylim])
                pyplot.tight_layout()
                pyplot.savefig(out_path)
                pyplot.close()

        out_txt = os.path.join(main_args.out, f"{rare_cluster}.overlap_genes.txt")
        with open(out_txt, "w") as f:
            for g in overlap_genes:
                f.write(f"{g}\n")

