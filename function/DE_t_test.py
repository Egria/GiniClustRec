import pandas
import argparse
import numpy
import os
import matplotlib_venn
from matplotlib import pyplot
from scipy import stats

from function.GiniClust_parameters import GiniClust_Parameters

def mean_in_log2space(x: pandas.Series) -> pandas.Series:
    return numpy.log2(numpy.mean(numpy.power(2.0, x) - 1) + 1)

def de_t_test(exprM_rawCounts_filter: pandas.DataFrame, rare_cells_list_all: dict, c_membership: pandas.Series, geneList_final:pandas.Index, main_args: argparse.Namespace, expr_params: GiniClust_Parameters):
    # for qPCR only
    for rare_cluster in rare_cells_list_all.keys():
        data_used3 = exprM_rawCounts_filter
        cells_1 = c_membership.index[c_membership == rare_cluster]
        cells_2 = c_membership.index[c_membership == "Cluster_1"]
        print(data_used3.loc[:, cells_1].columns)
        genes_use = data_used3.index
        mean_1 = data_used3.loc[genes_use, cells_1].apply(mean_in_log2space, axis=1)
        mean_2 = data_used3.loc[genes_use, cells_2].apply(mean_in_log2space, axis=1)
        total_diff = (mean_1 - mean_2).abs()
        genes_diff = total_diff[total_diff > expr_params.diff_cutoff].index.tolist()
        print(len(genes_use))
        p_vals = []
        for g in genes_use:
            group1 = data_used3.loc[g, cells_1].values
            group2 = data_used3.loc[g, cells_2].values
            stat, p = stats.ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
            p_vals.append(p)
        p_val = pandas.Series(p_vals, index=genes_use)
        log2fold_change = (mean_1 - mean_2).reindex(genes_use)
        mean_r = pandas.DataFrame({"mean.1":mean_1.reindex(genes_use), "mean.2":mean_2.reindex(genes_use)})

        differential_r = pandas.DataFrame({"mean.1":mean_1.reindex(genes_use),"mean.2":mean_2.reindex(genes_use),
                                           "p_val":p_val.reindex(genes_use), "log2fold_change":log2fold_change.reindex(genes_use)}, index=genes_use)
        differential_r = differential_r.sort_values(["p_val", "log2fold_change"], ascending=[True, False],
                                                    key = lambda col:numpy.abs(col) if col.name=="log2fold_change" else col)
        differential_r.loc[differential_r["p_val"] < 1e-22, "p_val"] = 1e-22
        cluster_diffgene = differential_r.index[(differential_r["p_val"]<1e-5)&(numpy.abs(differential_r["log2fold_change"])>numpy.log2(5.0))].tolist()
        print(len(cluster_diffgene))
        print(len(set(cluster_diffgene).intersection(geneList_final.tolist())))
        area1 = len(cluster_diffgene)
        area2 = len(geneList_final)
        n12 = len(set(cluster_diffgene).intersection(geneList_final.tolist()))
        cluster_overlap = numpy.array([[n12, area1-n12], [area2-n12, exprM_rawCounts_filter.shape[0]-(area1+area2-n12)]], dtype=int)
        print(cluster_overlap)
        oddsratio, ft_pvalue = stats.fisher_exact(cluster_overlap, alternative="two-sided")

        # Venn diagram
        labels = (f"diffgene\n{area1}", f"HighGiniGenes\n{area2}\n{ft_pvalue:.2e}")
        pyplot.figure(figsize=(6,6))
        matplotlib_venn.venn2(subsets=(area1-n12, area2-n12, n12), set_labels=labels, set_colors=("blue","yellow"), alpha=0.5)
        pyplot.savefig(os.path.join(main_args.out,"figures",f"{expr_params.experiment_id}_{rare_cluster}_diff_gene_overlap.pdf"))
        pyplot.close()

        # save results
        differential_r = differential_r.sort_values("log2fold_change", ascending=False)
        differential_r = differential_r.rename(columns={
            "mean.1":f"log2.mean.{rare_cluster}",
            "mean.2":"log2.mean.Cluster_1",
            "p_val":"t-test_pvalue",
            "log2fold_change":"log2fold_change"
        })
        differential_r.to_csv(os.path.join(main_args.out,f"{rare_cluster}.dff.gene.t-test.results.csv"), index=True)
        with open(os.path.join(main_args.out, f"{rare_cluster}.overlap_genes.txt"), 'w') as fh:
            for g in set(cluster_diffgene).intersection(geneList_final.tolist()):
                fh.write(str(g) + "\n")


