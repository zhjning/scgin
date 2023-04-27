import numpy as np
import pandas as pd
import multiprocessing 
import gc
import scginpy.utils
from scginpy.tool.unionRank import PartialRank 
from scipy.stats import rankdata 


def cellcount_to_rank_fast(cellexpr):
    rank_ind = np.where(cellexpr > 0)
    rank_res = np.repeat(0.0,len(cellexpr))
    rank_res[rank_ind] = rankdata(cellexpr[rank_ind], method = "average")
    return rank_res


def cal_copresent_cellnum(cellexpr1, cellexpr2):
    cind = np.intersect1d(np.where(cellexpr1 > 0)[0],np.where(cellexpr2 > 0)[0])
    return(len(cind))


def cal_netrankscore(rankscore1,rankscore2):
    return np.sqrt(rankscore1*rankscore2)


def generate_aggrankscore(exprmat):
    # exprmat = adata.X.toarray()
    
    rankpool = multiprocessing.Pool()
    cellranks = rankpool.map(cellcount_to_rank_fast, [exprmat[i,:] for i in range(exprmat.shape[0])])
    gc.collect()
    rankpool.close()
    rankpool.join()
    
    rank_result = []
    for i in range(exprmat.shape[0]):
        tmp_dict = pd.DataFrame(cellranks[1].tolist()).to_dict()[0]
        rank_result.append(tmp_dict)
    partial_rank = PartialRank()
    aggragated_ranks = partial_rank.aggregate_ranks(rank_result, method="borda", stat="geo")
    gc.collect()
    
    return aggragated_ranks

