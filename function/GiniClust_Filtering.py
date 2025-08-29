import argparse
import os
import pandas

from function.GiniClust_parameters import GiniClust_Parameters


def giniClust_filtering(exprM_raw_counts: pandas.DataFrame, main_args: argparse.Namespace, expr_params: GiniClust_Parameters) -> pandas.DataFrame:
    #main_args.expressed_cutoff:value lower than this value could be just noise.
    expressedinCell_per_gene = (exprM_raw_counts > expr_params.expressed_cutoff).sum(axis=1)
    nonMir = exprM_raw_counts.index[~exprM_raw_counts.index.str.contains("MIR|Mir", regex=True)]
    #because Mir gene is usually not accurate
    genelist = nonMir.intersection(exprM_raw_counts.index[expressedinCell_per_gene >= expr_params.minCellNum])
    expressedGene_per_cell = (exprM_raw_counts.loc[genelist, :] > 0).sum(axis=0)
    print(len(genelist))
    exprM_raw_counts_filter = exprM_raw_counts.loc[genelist, expressedGene_per_cell[expressedGene_per_cell >= expr_params.minGeneNum].index]
    print(exprM_raw_counts_filter.shape)
    exprM_raw_counts_filter.to_csv(os.path.join(main_args.out,f"{expr_params.experiment_id}_gene.expression.matrix.RawCounts.filtered.csv"), sep=',', index=True, header=True)
    return exprM_raw_counts_filter