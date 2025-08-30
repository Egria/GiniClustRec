import argparse
import os
import pandas
from typing import Dict
from function.GiniClust_parameters import GiniClust_Parameters
from function.GiniClust_Preprocess import giniClust_preprocess
from function.GiniClust_Filtering import giniClust_filtering
from function.GiniClust_Fitting import giniClust_fitting
from function.GiniClust_Clustering import giniClust_clustering

# Set work directory
workdir = os.getcwd()
assert os.path.isfile(os.path.join(workdir, "Library")) is False
if os.path.exists(os.path.join(workdir, "Library")):
    os.mkdir(os.path.join(workdir, "Library"))

# Command interface
parser = argparse.ArgumentParser(description='GiniClust',
                                 epilog='python GiniClust_Main.py -f sample_data/Data_GBM.csv -t RNA-seq -o GBM_results')
parser.add_argument("-f", "--file", required=True, type=str, help="Input dataset file name")
parser.add_argument("-t", "--type", required=True, type=str, help="Input dataset type: choose from 'qPCR' or 'RNA-seq' ")
parser.add_argument("-o", "--out", default="results", type=str, help="Output folder name [default=results]")
parser.add_argument("-e", "--epsilon", required=False, type=float, help="DBSCAN epsilon parameter qPCR:[default=0.25],RNA-seq:[default=0.5]")
parser.add_argument("-m","--minPts", required=False, type=int, help="DBSCAN minPts parameter qPCR:[default=5],RNA-seq:[default=3]")
args = parser.parse_args()

assert args.type in ["qPCR","RNA-seq"]
if args.type == "qPCR":
    if args.epsilon is None: args.epsilon = 0.25
    if args.minPts is None: args.minPts = 5
if args.type == "RNA-seq":
    if args.epsilon is None: args.epsilon = 0.5
    if args.minPts is None: args.minPts = 3

experiment_id = os.path.splitext(os.path.basename(args.file))[0]

######################## GiniClust Pipeline ########################

# Get parameters
params = GiniClust_Parameters(args)
params.set_experiment_id(experiment_id)

# Preprocess
exprM_results : Dict[str, pandas.DataFrame] = {}
exprM_results["raw"] = giniClust_preprocess(args, params)
exprM_rawCounts = exprM_results["raw"]

# Filtering
exprM_results["filter"] = giniClust_filtering(exprM_rawCounts, args, params)
exprM_rawCounts_filter = exprM_results["filter"]

# Gene selection
geneList_final = giniClust_fitting(exprM_rawCounts_filter, args, params)

# Clustering
cluster_results = giniClust_clustering(exprM_rawCounts_filter, args, params, geneList_final)
cell_cell_distance = cluster_results["cell_cell_dist"]
c_membership = cluster_results["c_membership"]
clustering_membership_r = cluster_results["clustering_membership_r"]
rare_cells_list_all = cluster_results["rare_cell"]

