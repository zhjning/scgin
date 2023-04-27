import gc
import os
import sys
import json
import torch
import pickle
import anndata
import logging
import pandas as pd
import numpy as np

from typing import List
from scipy.io import mmwrite

logger = logging.getLogger(__name__)

def check_gpu_conditions(devices: List[torch.device]):
    """check gpu conditions.
    """
    ### check occupations of current gpu devices ###
    if isinstance(devices, list):
        for dnum in range(len(devices)):
            print(torch.cuda.get_device_name(dnum) +": "+str(dnum))
            print("Memory Usage:")
            print("Allocated:", round(torch.cuda.memory_allocated(dnum)/1024**3,1), "GB")
            print("Cached:   ", round(torch.cuda.memory_cached(dnum)/1024**3,1), "GB")
            print()
    elif isinstance(devices, torch.device):
        print(torch.cuda.get_device_name(devices.index)+": "+str(devices.index))
        print("Memory Usuage:")
        print("Allocated:", round(torch.cuda.memory_allocated(devices.index)/1024**3,1), "GB")
        print("Cached:  ", round(torch.cuda.memory_cached(devices.index)/1024**3,1), "GB")


def check_cpu_conditions():
    """check cpu conditions.
    """
    import psutil
    virtual_msg = "virtual_memory used: " + str(round(psutil.virtual_memory().used*100/psutil.virtual_memory().total,2)) + "%"
    memory_msg = "current memory used: " + str(round(psutil.cpu_percent()*100,2)) + "%"
    print("Using CPU.")
    print(virtual_msg)
    print(memory_msg)


def check_memory_conditions(devices = None, show_message: bool = True, return_devices : bool = False):
    """check memory conditions.
    """
    if devices == None:
        return_devices = True
        devices = [] if torch.cuda.is_available() else 'cpu'
        if isinstance(devices, list):
            device_num = torch.cuda.device_count()
            for dnum in range(device_num):
                devices.append(torch.device(dnum))    
    if isinstance(devices, list):
        if isinstance(devices[0], torch.device):
            logger.info("Using GPUs.")
            if show_message:
                check_gpu_conditions(devices)
        else:
            logger.error("Only CPU or GPU devices are allowed for devices.")
    elif isinstance(devices, torch.device):
        logger.info("Using GPU.")
        if show_message:
            check_gpu_conditions(devices)
    elif devices == "cpu":
        logger.info("Using CPU.")
        if show_message:
            check_cpu_conditions()
    else:
        logger.error("Only CPU or GPU devices are allowed for devices.")
    if return_devices:
        return devices


def release_memory_occupation():
    """release memory occupation.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_file(filepath: str, **kwargs):
    """load file.
    """
    dirpath = os.path.dirname(filepath)
    filename0 = os.path.basename(filepath)
    filename, ext = os.path.splitext(filename0)
    if not os.path.exists(filepath):
        logger.error(filepath + " is not exists, please check your import file path.")
    else:
        if ext == ".pickle":
            with open(filepath, "rb") as f:
                loaded = pickle.load(f)            
        elif ext == ".csv" or ext == ".txt" or ext == ".tsv":
            if "delimiter" not in kwargs.keys(): 
                if ext == ".csv":
                    delimiter = ","
                elif ext == ".txt" or ext == ".tsv":
                    delimiter = '\t'
                loaded = pd.read_csv(filepath, delimiter = delimiter, **kwargs)
            else:
                loaded = pd.read_csv(filepath, **kwargs)
        elif ext == ".xlsx" or ext == ".xls":
            loaded = pd.read_excel(filepath, **kwargs)
        elif ext == ".json":
            with open(filepath, "rb") as f:
                loaded = json.load(f, **kwargs)
        elif ext == ".loom":
            loaded = anndata.read_loom(filepath, **kwargs)
        elif ext == ".h5ad":
            loaded = anndata.read_h5ad(filepath, **kwargs)
        elif ext == ".h5": ## hdf format
            loaded = anndata.read_hdf(filepath, **kwargs)
        elif ext == ".mtx":
            loaded = anndata.read_mtx(filepath, **kwargs)
        elif ext == ".gz":
            try:
                loaded = anndata.read_umi_tools(filepath, **kwargs)
            except ValueError:
                logger.error(filepath + " is not gzipped condensed count matrix, umi_tools was tried for gzipped files.")
                raise
        elif ext == '':
            try:
                loaded = pd.read_csv(filepath, **kwargs)
            except ValueError:
                logger.error(filepath + " format is not identified, default format is tab-separated table file, please check your data format.")
                raise
    return loaded


def save_file(data, filepath: str, data_format = ".pickle", **kwargs):
    """save file.
    """
    ## check filepath validation
    if os.path.exists(filepath):
        logger.warning(filepath + " already exists, the orginal file will be overwritten.")
    dirpath = os.path.dirname(filepath)
    os.makedirs(dirpath, exist_ok = True)
    filename0 = os.path.basename(filepath)
    filename, ext = os.path.splitext(filename0)

    ## define acceptable data format and file delimiters
    avail_dformat_list = [".pickle", ".csv", ".tsv", ".txt", ".h5ad", ".h5", ".json", ".xlsx", ".xls"]
    avail_delimiters = dict({".csv":",", ".txt":"\t", "tsv":"\t"})

    ## check data format
    if data_format in avail_dformat_list:
        dformat = data_format
    else:
        logger.warning(data_format + "is not available, use default data format: "+avail_dformat_list[0])
        dformat = avail_dformat_list[0]
    if ext != dformat:
            logger.error(filepath + "'s suffix not matches output data format, convert filepath suffix to "+dformat)
            filepath = dirpath + os.sep + filename + dformat    

    ## save to target data format
    if dformat == ".pickle":
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif dformat == '.csv' or dformat == '.tsv' or dformat == '.txt':
        delimiter = avail_delimiters[dformat]
        if isinstance(data, pd.DataFrame):
            if 'index' in kwargs.keys():
                if 'sep' in kwargs.keys():
                    data.to_csv(filepath, **kwargs)
                else:
                    data.to_csv(filepath, sep=delimiter, **kwargs)
            else:
                data.to_csv(filepath, index=False, sep=delimiter, **kwargs)
    elif dformat == ".h5ad":
        if isinstance(data, anndata.AnnData):
            data.write_h5ad(filepath, **kwargs)
        else:
            logger.warning("only anndata object can be saved to h5ad, please transform object to anndata object before this step")
    elif dformat == ".h5":
        if isinstance(data, pd.DataFrame):
            if 'index' in kwargs.keys():
                data.to_hdf(filepath, **kwargs)
            else:
                data.to_hdf(filepath, index=False, **kwargs)
        else:
            logger.warning("input data should be pd.DataFrame object, please transform object to pd.DataFrame object before this step")
    elif dformat == ".json":
        with open(filepath, 'wb') as f:
            json.dump(data, f)
    elif dformat == ".mtx":
        try:
            mmwrite(filepath, data, **kwargs)
        except ValueError:
            logger.warning("input data should be sparse or dense 2-D array")
    elif dformat == ".xls" or dformat == ".xlsx":
        if isinstance(data, pd.DataFrame):
            if 'index' in kwargs.keys():
                data.to_excel(filepath, **kwargs)
        else:
            logger.warning("input data should be pd.DataFrame object, please transform object to pd.DataFrame object before this step")


## updated 20210301

# def check_genepairs(ggpairs1, ggpairs2):
#     """check genepairs.
#     """
#     # common_ggpairs = list(set(ggpairs1).intersection(ggpairs2))
#     # ggpairs_indice = np.array([np.array([ggpairs1.index(common_ggpairs[i]), ggpairs2.index(common_ggpairs[i])]) for i in range(len(common_ggpairs))])
#     if isinstance(ggpairs1, pd.DataFrame) and isinstance(ggpairs2, pd.DataFrame):
#         a = ggpairs1.reset_index().set_index(['gind1','gind2'])
#         b = ggpairs2.reset_index().set_index(['gind1','gind2'])
#         alen = a.shape[0]
#         blen = b.shape[0]
#         if alen == blen:
#             if (a.index == b.index).all():
#                 # ggpairs_indice = np.concatenate([[ggpairs1.iloc[:,0].tolist()],[ggpairs2.iloc[:,0].tolist()]],axis=0).transpose()
#                 return pd.concat([a, b], axis=1).to_numpy()
#             else:
#                 common_ggpairs = np.intersect1d(a.index, b.index)
#                 aa = a.loc[common_ggpairs]
#                 bb = b.loc[common_ggpairs]
#                 return pd.concat([aa, bb], axis=1).to_numpy()
#         else:
#             common_ggpairs = np.intersect1d(a.index, b.index)
#             aa = a.loc[common_ggpairs]
#             bb = b.loc[common_ggpairs]
#             return pd.concat([aa, bb], axis=1).to_numpy()
#     else:
#         logger.error("ggpairs should be pd.DataFrame object")
#         return None


# def generate_genename_indice(genename_list: list, genename_filepath: str, if_return: bool = False):
#     """generate genename indice.
#     """
#     if isinstance(genename_list, list):
#         if len(genename_list) > 0:
#             genename_annotation = pd.DataFrame(genename_list, index = np.arange(len(genename_list)), columns=['Gene'])
#         if os.path.exists(genename_filepath):
#             logger.warning(genename_filepath + " already exists, original file will be overwritten.")
#             save_file(genename_annotation, genename_filepath)
#     else:
#         logger.error("genename list should be list")
#     if if_return:
#         return genename_annotation
#     else:
#         return None


# def generate_genepair_indice(genepairname_list: list, genepairname_filepath: str, if_return: bool = False):
#     """generate genepair indice.
#     """
#     if isinstance(genepairname_list, list):
#         if len(genepairname_list) > 0:
#             genepairname_annotation = pd.DataFrame(genepairname_list, index = np.arange(len(genepairname_list)), columns=['Genepair_1','Genepair_'])
#         if os.path.exists(genepairname_filepath):
#             logger.warning(genepairname_filepath + " already exists, original file will be overwritten.")
#             save_file(genepairname_annotation, genepairname_filepath)
#     else:
#         logger.error("genepair name list should be list")
#     if if_return:
#         return genepairname_annotation
#     else:
#         return None
    

# def get_genename_annotation(genename_filepath:str, gene_indice = None):
#     """get genename annotation.
#     """
#     genename_annotation = load_file(genename_filepath)
#     if isinstance(genename_annotation, pd.DataFrame):
#         if gene_indice == None:
#             return genename_annotation
#         else:
#             genename_list = genename_annotation.iloc[gene_indice,0].tolist()
#             return genename_list
#     else:
#         logger.error("format of gene names' annotation table is wrong, please check the original file.")


# def get_genepairname(genename_annotation_path: str, ggpair_indices):
#     """get genepairname.
#     """
#     if isinstance(ggpair_indices, pd.DataFrame):
#         genepair_names = [tuple(get_genename_annotation(genename_annotation_path,list(i))) for i in ggpair_indices.reset_index().set_index(['gind1','gind2']).index.tolist()]
#         return genepair_names
#     elif isinstance(ggpair_indices, tuple):
#         genepair_names = [tuple(get_genename_annotation(genename_annotation_path,list(i)))]
#         return genepair_names
#     else:
#         logger.error("genepair_indices is not identified, please use pd.DataFrame or tuple that contains genepair indices")
#         return None


# def get_genepair_index_annotation(genepair_index_filepath:str, genepair_indice = None):
#     """get genepair index annotation.
#     """
#     genepair_index_annotation = load_file(genepair_index_filepath)
#     if isinstance(genepair_index_annotation, pd.DataFrame):
#         if genepair_indice == None:
#             return genepair_index_annotation
#         else:
#             genepair_index_list = genepair_index_annotation.reset_index().set_index(['gind1','gind2']).index[genepair_indice].tolist()
#             return genepair_index_list
#     else:
#         logger.error("format of genepair indices' annotation table is wrong, please check the original file.")


# def get_genepairname_annotation(genepair_indices,
#                                 genename_annotation = None,
#                                 genename_filepath = None, 
#                                 genepair_index_annotation = None, 
#                                 genepair_index_filepath = None):
#     """get genepairname annotation.
#     """
#     if genename_annotation == None:
#         if genename_filepath == None:
#             logger.error("genename annotation table/filepath is not found")
#             return None
#         else:
#             genename_annotation = load_file(genename_filepath)
#     if genepair_index_annotation == None:
#         if genepair_index_filepath == None:
#             logger.warning("genepair index annotation table/filepath is not found, using genename annotations.")
#         else:
#             genepair_index_annotation = load_file(genepair_index_filepath)    
#     if not isinstance(genename_annotation, pd.DataFrame):
#         logger.error("gene name annotation shoud be a pandas DataFrame object")
#         return None
#     if not isnone(genepair_index_annotation):
#         if not isinstance(genepair_index_annotation, pd.DataFrame):
#             logger.error("genepair index annotation should be a pandas DataFrame object")
#             return None
#     return get_genepairname2(genename_annotation = genename_annotation, 
#                             genepair_index_annotation = genepair_index_annotation,
#                             genepair_indices = genepair_indices)


# def get_genepairname2(genename_annotation: pd.DataFrame, ggpair_index_annotation, ggpair_indices):
#     if isinstance(ggpair_indices, list):
#         genepair_names = []
#         for ggpair_index in ggpair_indices:
#             if not isinstance(ggpair_index, int):
#                 logger.error("input ggpair indice/index is not identified, please use list of intergers or intergers")
#                 return None
#             genepair_names.append((genename_annotation.iloc[ggpair_index_annotation.iloc[ggpair_index, 0], 0],
#             genename_annotation.iloc[ggpair_index_annotation.iloc[ggpair_index, 1], 0]))
#         return genepair_names
#     elif isinstance(ggpair_indices, int):
#         return (genename_annotation.iloc[ggpair_index_annotation.iloc[ggpair_indices, 0], 0],
#             genename_annotation.iloc[ggpair_index_annotation.iloc[ggpair_indices, 1], 0])
#     else:
#         logger.error("input ggpair indice/index is not identified, please use list of intergers or intergers")


# def isnone(obj):
#     isnone = True
#     if not isinstance(obj, type(None)):
#         isnone = False
#     return isnone


