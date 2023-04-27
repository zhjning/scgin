# Tabular Muris, application
from scginpy.utils import load_file, save_file
import scginpy
import scanpy as sc
import numpy as np
import os

## import pretreated data by scanpy (python) and seurat (R)
data_filepath = '/home/zhjning/workspace/datastore/TM/TM_droplet_mat.annotated.h5ad'
data = sc.read_h5ad(data_filepath)

## If cells are classified to different clusters, set a group variable before setting the feature variable,
## if there is no pre-defined group, use same value for the group variable and feature variable.
group_feature = 'tissue' 

feature = 'cell_ontology_class'
feature_val_lab = 'B cell'
feature_val_list = ['B cell','Fraction A pre-pro B cell','early pro-B cell','immature B cell','late pro-B cell']

output_dir = "/home/zhjning/workspace/projects/scgin_2021/"+ os.sep + feature_val_lab + os.sep
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

subdata = data[np.where(data.obs[feature].isin(feature_val_list))]

group_flist = np.unique(subdata.obs.loc[:,group_feature])

group_net = []
# for i in range(len(group_flist)):
for group_i in range(len(group_flist)):
    adata = subdata[np.where(np.isin(subdata.obs.loc[:,group_feature], group_flist[group_i]))]
    group_na_expr = np.where(np.sum(adata.X.toarray(),axis=0)>0)[0]
    adata = adata[:, group_na_expr]
    anet = scginpy.scgin.network_wrapper(adata, output_dir, label=str(group_flist[group_i]), mode="torch", if_return=True)
    save_file(anet, output_dir + os.sep + str(group_flist[group_i]) + "_net.pickle")
    group_net.append(anet)

## 20210302, load B cell network in different tissues:
feature_val_lab = 'B cell'
group_val_lab = 'Lung'

filepath = "/home/zhjning/workspace/projects/scgin_2021/"+ os.sep + feature_val_lab + os.sep + group_val_lab + "_net.pickle" 
anet = load_file(filepath)

