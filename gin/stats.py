
import scginpy.utils
import multiprocessing
import numpy as np
import pandas as pd
import scanpy as sc
import tempfile
import torch
import math
import gc
import os

import dask.dataframe as dd
import scipy.stats as stats

from scginpy.tool.unionRank import PartialRank
from scginpy.utils import save_file, load_file
from scipy.stats import rankdata



## rank cell count
def cellcount_to_rank_fast(cellexpr):
    rank_ind = np.where(cellexpr > 0)
    rank_res = np.repeat(0.0,len(cellexpr))
    rank_res[rank_ind] = rankdata(cellexpr[rank_ind], method = "average")
    return rank_res


## coccur cell number between two genes in a cell population
def cal_copresent_cellnum(cellexpr1, cellexpr2):
    cind = np.intersect1d(np.where(cellexpr1 > 0)[0],np.where(cellexpr2 > 0)[0])
    return(len(cind))


## average rankscore between two genepairs (rankscore number smaller, expression higher)
def cal_netrankscore(rankscore1, rankscore2):
    return np.sqrt(rankscore1*rankscore2)


## genepair-feature, partial rank score
def generate_aggrankscore(adata):
    exprmat = adata.X.toarray()
    
    rankpool = multiprocessing.Pool()
    cellranks = rankpool.map(cellcount_to_rank_fast, [exprmat[i,:] for i in range(exprmat.shape[0])])
    gc.collect()
    rankpool.close()
    rankpool.join()
    
    rank_result = []
    for i in range(adata.shape[0]):
        tmp_dict = pd.DataFrame(cellranks[1].tolist()).to_dict()[0]
        rank_result.append(tmp_dict)
    PLRA = PartialRank()
    aggregated_ranks = PLRA.aggregate_ranks(rank_result, method="borda", stat="geo")
    gc.collect()
    
    return aggregated_ranks


## genepair-feature, cooccurscore
def network_cooccurscore(adata, ggpairs):
    exprmat = adata.X.toarray()
    cooccurscore = [cal_copresent_cellnum(exprmat[:,ggpair[0]], exprmat[:,ggpair[1]]) for ggpair in ggpairs]
    return cooccurscore


## genepair-rank score.
def network_rankscore(aggRanks, ggpairs):
    rankscore = aggRanks[0]
    paired_rankscores = [cal_netrankscore(rankscore[ggpair[0]],rankscore[ggpair[1]]) for ggpair in ggpairs]
    return paired_rankscores


## genepair-feature, generator
def generator_for_cooccurscore(adata, ggpairs):
    exprmat = adata.X.toarray()
    for ggpair in ggpairs:
        yield exprmat[:,ggpair[0]], exprmat[:,ggpair[1]]


## genepair-feature, generator
def generator_for_rankscore(aggRanks, ggpairs):
    rankscore = aggRanks[0]
    for ggpair in ggpairs:
        yield rankscore[ggpair[0]], rankscore[ggpair[1]]


## genepair-feature
def network_cooccurscore_p(adata, ggpairs):
    ggpair_pool = multiprocessing.Pool(processes = multiprocessing.cpu_count()-2)
    cooccurscore = ggpair_pool.starmap(cal_copresent_cellnum, generator_for_cooccurscore(adata, ggpairs))
    ggpair_pool.close()
    ggpair_pool.join()
    return cooccurscore


## genepair-feature
def network_rankscore_p(aggRanks, ggpairs):
    ggpair_pool = multiprocessing.Pool(processes = multiprocessing.cpu_count()-2)
    paired_rankscores = ggpair_pool.starmap(cal_netrankscore, generator_for_rankscore(aggRanks, ggpairs))
    ggpair_pool.close()
    ggpair_pool.join
    return paired_rankscores


## genepair-class, optimazation paired speed 
def prior_network_generator(adata, mode="torch", ggpairs=[], aggRanks=None, chunksize = 10000, clear_temp = True):
    
    net_factor = adata.shape[0]*adata.shape[1]
    
    if aggRanks is None:
        aggRanks = generate_aggrankscore(adata)
    
    ggpair_net = pd.DataFrame()
    if len(ggpairs)==0:
        ggpair_net = pd.DataFrame(torch.combinations(torch.tensor(np.arange(adata.shape[1])), r=2).numpy(), columns = ['gind1','gind2'])
        ggpairs = ggpair_net.reset_index().set_index(['gind1','gind2']).index.tolist()
    elif len(ggpairs) > 0:
        ggpair_net = pd.DataFrame(ggpairs, columns=['gind1','gind2'])
    ## slow mode 
    if mode == "normal":
        ggpair_net['cooccurscore'] = network_cooccurscore(adata, ggpairs)
        ggpair_net['rankscore'] = network_rankscore(aggRanks, ggpairs)
        ## accelerated code chunk
        ggpair_net['activity'] = dd.from_pandas(ggpair_net, npartitions=2*multiprocessing.cpu_count())\
                                    .map_partitions(lambda df: df.apply(lambda x: x.cooccurscore*x.rankscore/net_factor, axis=1))\
                                    .compute(scheduler="processes")
    ## multi-threads mode
    elif mode == "multiprocessing":
        # start a multiprocessing pool
        ggpair_net['cooccurscore'] = network_cooccurscore_p(adata, ggpairs)
        ggpair_net['rankscore'] = network_rankscore_p(aggRanks, ggpairs)
        ggpair_net['activity'] = dd.from_pandas(ggpair_net, npartitions=2*multiprocessing.cpu_count())\
                                                .map_partitions(lambda df: df.apply(lambda x: x.cooccurscore*x.rankscore/net_factor, axis=1))\
                                                .compute(scheduler="processes")
    ## torch mode
    elif mode == "torch":
        ggpairs_size = len(ggpairs)
        tmpdir = tempfile.mkdtemp()
        print("tempfile location:"+os.sep+tmpdir)
        if ggpairs_size > chunksize:
            ggpairs_chunks = math.ceil(ggpairs_size/chunksize)
        else:
            ggpairs_chunks = 1
        for ggpairs_chunk in range(ggpairs_chunks):
            if ggpairs_chunk != ggpairs_chunks - 1:
                tmp_ggpairs = ggpairs[ggpairs_chunk*chunksize : (ggpairs_chunk+1)*chunksize]
                tmp_ggpair_net = ggpair_net.iloc[ggpairs_chunk*chunksize : (ggpairs_chunk+1)*chunksize, :]
            else:
                tmp_ggpairs = ggpairs[ggpairs_chunk*chunksize : ggpairs_size]
                tmp_ggpair_net = ggpair_net.iloc[ggpairs_chunk*chunksize : ggpairs_size, :]
            # using torch.tensor for calculation
            exprmat = torch.tensor(adata.X.toarray())
            exprmat[exprmat > 0] = 1
            exprmat_co = torch.tensor(exprmat[:, [ggpair[0] for ggpair in tmp_ggpairs]])
            exprmat_co = exprmat_co + torch.tensor(exprmat[:, [ggpair[1] for ggpair in tmp_ggpairs]]) -1
            del exprmat
            gc.collect()
            exprmat_co[exprmat_co < 1] = 0
            tmp_ggpair_net['cooccurscore'] = torch.sum(exprmat_co,axis=0).numpy()
            del exprmat_co
            gc.collect()
            # ggpair_net['cooccurscore'] = [len(np.intersect1d(b[0][torch.where(b[1]==ggpair_net.gind1[i])[0]],
            #                b[0][torch.where(b[1]==ggpair_net.gind2[i])[0]])) for i in range(ggpair_net.shape[0])]
            a = torch.tensor([aggRanks[0][i] for i in tmp_ggpair_net.iloc[:,0]])
            b = torch.tensor([aggRanks[0][i] for i in tmp_ggpair_net.iloc[:,0]])
            c = np.sqrt(a*b)
            del a, b
            gc.collect()
            tmp_ggpair_net['rankscore'] = c.numpy()
            del c
            gc.collect()
            c = torch.tensor(tmp_ggpair_net.loc[:,['cooccurscore','rankscore']].to_numpy())
            tmp_ggpair_net['activity'] = c[:,0]*c[:,1]/net_factor
            
            # ggpairs_subnets = pd.concat([ggpairs_subnets, tmp_ggpair_net], axis=0)
            save_file(tmp_ggpair_net, tmpdir + os.sep + "tmp_ggpair_net." + str(ggpairs_chunk) + ".pickle")
            del tmp_ggpair_net
            gc.collect()
        
        ggpair_net = pd.DataFrame()
        for ggpairs_chunk in range(ggpairs_chunks):
            tmp_ggpair_net = load_file(tmpdir + os.sep + "tmp_ggpair_net." + str(ggpairs_chunk) + ".pickle")
            ggpair_net = pd.concat([ggpair_net, tmp_ggpair_net], axis=0)
            del tmp_ggpair_net
            gc.collect()
            if clear_temp is True:
                os.remove(os.path.join(tmpdir, "tmp_ggpair_net." + str(ggpairs_chunk) + ".pickle"))
    else:
        print("Mode " + mode + " is not identified.")
    return ggpair_net

## genepair-class
def network_wrapper(adata, output_dir, label="adata", mode = "torch", if_return=True, if_filter=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    anet = prior_network_generator(adata, mode = mode, chunksize=300000)
    save_file(anet, output_dir+os.sep+label+"_net.pickle")
    gc.collect()

    anet_f = None
    if if_filter:
        anet_f = anet.iloc[np.where(anet['activity']>0.1)[0],].sort_values('activity',ascending=False)
        save_file(anet_f, output_dir+os.sep+label+"_net.filtered.pickle")
        del anet
        gc.collect()
    
    if if_return:
        if if_filter:
            return anet_f
        else:
            return anet
    else:
        return None

 ## genepair-class   
def ggpair_index2ggpair_geneNames(ggpair_index, refgenes, refname = 'ORF'):
    df = pd.DataFrame([refgenes.loc[pd.DataFrame(index=ggpair_index).reset_index().iloc[:,0],refname].tolist(),
                       refgenes.loc[pd.DataFrame(index=ggpair_index).reset_index().iloc[:,1],refname].tolist()]).transpose()
    return df

