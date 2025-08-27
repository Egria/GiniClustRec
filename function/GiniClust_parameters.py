import argparse

class GiniClust_Parameters():
    def __init__(self, main_args: argparse.Namespace):
        assert main_args.type in ['RNA-seq', 'qPCR']
        self.minCellNum : int = 0                     # filtering, for at least expressed in how many cells
        self.minGeneNum : int = 0                     # filtering, for at least expressed in how many genes
        self.expressed_cutoff : int = 0               # filtering, for raw counts
        self.log2_expr_cutoffl : float = 0.0          # cutoff for range of gene expression
        self.log2_expr_cutoffh : float = 0.0          # cutoff for range of gene expression
        self.gini_pvalue_cutoff : float = 0.0         # fitting, Pvalue, control how many gene finally used
        self.norm_gini_cutoff : float = 0.0           # fitting, NomGini, control how many gene finally used, 1 means not used
        self.span : float= 0.0                        # parameter for LOESS fitting
        self.outlier_remove : float = 0.0             # parameter for LOESS fitting
        self.gamma : float = 0.0                      # parameter for clustering
        self.diff_cutoff : float = 0.0                # MAST analysis, filter gene don't have high log2_foldchange to reduce gene num
        self.lr_p_value_cutoff : float = 0.0          # MAST analysis, pvalue cutoff to identify differential expressed gene
        self.countsForNormalized : int = 0
        self.rare_p : float = 0.0                     # proposition of cell number < this value will be considered as rare cell clusters
        self.perplexity : int = 0
        self.eps = main_args.epsilon                  # parameter for DBSCAN
        self.minPts : int = main_args.minPts          # parameter for DBSCAN
        self.experiment_id : str = ""
        self.fill_args(main_args.type)

    def fill_args(self, data_type):
        assert data_type in ['RNA-seq', 'qPCR']
        if data_type == 'RNA-seq':
            self.minCellNum = 3
            self.minGeneNum = 2000
            self.expressed_cutoff = 1
            self.log2_expr_cutoffl = 0.0
            self.log2_expr_cutoffh = 20.0
            self.gini_pvalue_cutoff = 0.0001
            self.norm_gini_cutoff = 1.0
            self.span = 0.9
            self.outlier_remove = 0.75
            self.gamma = 0.9
            self.diff_cutoff = 1.0
            self.lr_p_value_cutoff = 1e-5
            self.countsForNormalized = 100000
            self.rare_p = 0.05
            self.perplexity = 30
        elif data_type == 'qPCR':
            self.minCellNum = 3
            self.minGeneNum = 30
            self.expressed_cutoff = 1
            self.log2_expr_cutoffl = 1.0
            self.log2_expr_cutoffh = 24.0
            self.gini_pvalue_cutoff = 1.0
            self.norm_gini_cutoff = 0.05
            self.span = 0.9
            self.outlier_remove = 0.75
            self.gamma = 0.9
            self.diff_cutoff = 1.0
            self.lr_p_value_cutoff = 1e-5
            self.countsForNormalized = 100000
            self.rare_p = 0.05
            self.perplexity = 20
        else:
            pass

    def set_experiment_id(self, experiment_id: str):
        self.experiment_id = experiment_id



