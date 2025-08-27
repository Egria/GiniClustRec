import os
import argparse
import pandas
import numpy
from pandas import DataFrame

from .GiniClust_parameters import GiniClust_Parameters

def giniClust_preprocess(main_args: argparse.Namespace, expr_params: GiniClust_Parameters) -> pandas.DataFrame:
    subdir = [main_args.out, main_args.out + '/figures']
    for s in subdir:
        if not os.path.isdir(s):
            os.mkdir(s)
    if (main_args.type == 'RNA-seq'):
        exprM_raw_counts = pandas.read_csv(main_args.file, sep=',', header=0)
        exprM_raw_counts.to_csv(os.path.join(main_args.out, f"{expr_params.experiment_id}_rawCounts.csv"), sep=',', index=True, header=True, quoting=0)
        return exprM_raw_counts
    elif (main_args.type == 'qPCR'):
        exprM_log2 = pandas.read_csv(main_args.file, sep=',', header=0)
        exprM_nor = pandas.DataFrame(numpy.power(2, exprM_log2) - 1)
        exprM_raw_counts = exprM_nor
        exprM_raw_counts.to_csv(os.path.join(main_args.out, f"{expr_params.experiment_id}_rawCounts.csv"), sep=',', index=True, header=True, quoting=0)
        return exprM_raw_counts
    else:
        return None


