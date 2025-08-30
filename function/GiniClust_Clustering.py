import argparse
import numpy
import pandas
import warnings
import os
from scipy import sparse
from sklearn.cluster import DBSCAN
from function.GiniClust_parameters import GiniClust_Parameters


def jaccard(m: pandas.DataFrame) -> pandas.DataFrame:
    row_names = m.index
    mat = sparse.csr_matrix((m.values > 0).astype(numpy.float64))
    a = mat @ mat.T
    a = a.tocoo()
    b = numpy.asarray(mat.sum(axis=1)).ravel()
    denom = b[a.row] + b[a.col] - a.data
    vals = a.data / denom
    j = sparse.coo_matrix((vals, (a.row, a.col)), shape=a.shape)
    j_dense = j.toarray()
    return pandas.DataFrame(j_dense, index=row_names, columns=row_names)


def giniClust_clustering(exprM_raw_counts_filter:pandas.DataFrame, main_args:argparse.Namespace, expr_params:GiniClust_Parameters, genelist_final:pandas.Index) -> dict:
    if main_args.type == "RNA-seq":
        m = exprM_raw_counts_filter.loc[genelist_final, :]
        m2 = m.copy()
        bc_list_low = []
        bc_list_med = []
        bc_list_high = []
        for rn in range(m.shape[0]):
            # if t is the expression vector, the question asked here is which count value x is the smallest one, when sum(t[t>x]) > Gamma % * sum(t).
            t = m.iloc[rn, :].to_numpy(dtype=int)
            t_table = pandas.Series(t).value_counts().sort_index()
            c = t_table.index.to_numpy(dtype=int)
            f = t_table.to_numpy(dtype=int)
            #significant difference from original R code
            denom = t[t > 0].sum()
            if denom == 0:
                csum = numpy.zeros_like(c, dtype=float) #all zero counts
                warnings.warn("Gene selected with all zero counts")
            else:
                contrib = c * f
                tail_ge = numpy.cumsum(contrib[::-1])[::-1]
                csum = tail_ge / denom
            order_desc = numpy.argsort(c)[::-1]
            c_desc = c[order_desc]
            f_desc = f[order_desc]
            csum_desc = csum[order_desc]
            hits = numpy.where(csum_desc > expr_params.gamma)[0]
            if hits.size > 0:
                n_idx = int(hits[0])
            else:
                n_idx = 0
            n_idx = max(2, n_idx)
            if n_idx >= len(c_desc) - 1:
                n_idx = max(0, len(c_desc) - 2)
            bc_list_high.append(float(c_desc[n_idx]))
            bc_list_low.append(float(c_desc[n_idx + 1]))
            bc_list_med.append(0.5 * (bc_list_high[-1] + bc_list_low[-1]))
        top_n_gene = max(int(len(bc_list_low)*0.1), 10)
        rawCounts_cutoff = int(numpy.floor(numpy.mean(bc_list_med[:top_n_gene])))

        # binarization
        all_gene_as_col_exprM_rawCounts = exprM_raw_counts_filter.T.copy()
        all_gene_as_col_exprM_rawCounts_binary = (all_gene_as_col_exprM_rawCounts >= rawCounts_cutoff).astype(int)
        final_gene_as_col_exprM_rawCounts_binary = all_gene_as_col_exprM_rawCounts_binary[all_gene_as_col_exprM_rawCounts_binary.columns.intersection(genelist_final)]
        print(final_gene_as_col_exprM_rawCounts_binary.shape)

        # locate cells whose gene expression values are all zeros
        index_cell_zero = final_gene_as_col_exprM_rawCounts_binary.index[(final_gene_as_col_exprM_rawCounts_binary > 0).sum(axis=1) == 0]
        cell_cell_jaccard_distance = 1 - jaccard(final_gene_as_col_exprM_rawCounts_binary)

        # convert distance between two 'zero cells' to zero
        for cell in index_cell_zero:
            cell_cell_jaccard_distance.loc[cell, index_cell_zero] = 0
            cell_cell_jaccard_distance.loc[index_cell_zero, cell] = 0
        print(cell_cell_jaccard_distance.shape)

        # dbscan
        title = f"eps.{expr_params.eps}.MinPts.{expr_params.minPts}"
        db = DBSCAN(eps=expr_params.eps, min_samples=expr_params.minPts, metric="precomputed")
        labels = db.fit_predict(cell_cell_jaccard_distance.to_numpy())
        data_mclust = pandas.DataFrame({"Cell": cell_cell_jaccard_distance.index, "Cluster": labels}).set_index("Cell")

        #rename cluster names based on the size of each cluster
        o_membership = ["db_" + str(c) for c in data_mclust["Cluster"]]
        o_membership = pandas.Series(o_membership, index=data_mclust.index)
        o_membership = o_membership.replace({"db_-1": "Singleton"})
        c_membership = o_membership.copy()
        cluster_stat = c_membership.value_counts().reset_index()
        cluster_stat.columns = ["o_membership", "Freq"]
        cluster_stat = cluster_stat[cluster_stat["o_membership"] != "Singleton"]
        cluster_stat = cluster_stat.sort_values("Freq", ascending=False).reset_index(drop=True)
        cn = cluster_stat.shape[0]
        cluster_stat["new_membership"] = [f"Cluster_{i+1}" for i in range(cn)]
        mapping = dict(zip(cluster_stat["o_membership"], cluster_stat["new_membership"]))
        c_membership = c_membership.replace(mapping)
        cluster_stat = c_membership.value_counts().reset_index()
        cluster_stat.columns = ["Cluster", "Freq"]
        cluster_id = pandas.DataFrame({"Cell_ID": cell_cell_jaccard_distance.index, "GiniClust_membership": c_membership.astype(str).values})
        print(cluster_id["GiniClust_membership"].value_counts())
        print(c_membership.value_counts())

        # if a cluster smaller than 5% of the total cell number, we call it as a rare cell types cluster.
        cell_num = exprM_raw_counts_filter.shape[1] * expr_params.rare_p
        rare_cells_list_all = {}
        for c in pandas.unique(c_membership):
            cells_in_c = cell_cell_jaccard_distance.index[c_membership == c]
            if (len(cells_in_c) < cell_num) and (c != "Singleton"):
                rare_cells_list_all[c] = list(cells_in_c)
        rare_cluster_names = list(rare_cells_list_all.keys())
        print(rare_cluster_names)

        clustering_membership_r = pandas.DataFrame({"cell.ID": cell_cell_jaccard_distance.index.astype(str), "cluster.ID": c_membership.astype(str).values})
        clustering_membership_r.to_csv(os.path.join(main_args.out, f"{expr_params.experiment_id}_clusterID.csv"), index=False)
        with open(os.path.join(main_args.out, f"{expr_params.experiment_id}_rare_cells_list.txt"), "w") as fh:
            for cluster_name, cells in rare_cells_list_all.items():
                fh.write(f"{cluster_name} :\n")
                if cells:
                    for i in range(0, len(cells), 20):
                        line = ", ".join(map(str, cells[i:i+20]))
                        fh.write(line + "\n")
                fh.write("\n")
        retdict = {"cell_cell_dist": cell_cell_jaccard_distance, "c_membership": c_membership,
                   "clustering_membership_r": clustering_membership_r, "rare_cell": rare_cells_list_all}
        return retdict

    elif main_args.type == "qPCR":
        pass
    else:
        raise ValueError("Unknown type")