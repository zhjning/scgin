# -*- coding: utf-8 -*-

import os
import sys
import torch
import random
import datetime
import pickle
import tempfile
import logging
import anndata

import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import combinations
from functools import reduce
from typing import Tuple, Dict
from collections import UserList
from scipy.stats import stats

from scginpy.gin.stats import compute_rank_correlation_2d
from scginpy.gin.smo import SMO
from scginpy.utils import *

torch.backends.cudnn.benchmark = True
logger = logging.getLogger('scgin')



## from scginpy import scgin
class scGIN():
    """scGIN object.

    A class of single-cell gene interaction network.

    Args:
        project_name: basestring
            the project name of scGIN object.
        data: basestring
            the data included single-cell raw data (cellxgene) for GIN generation.
        data_path: basestring
            complete filepath of the data that is going to be loaded.
        useCPU: boolean
            default False. If useCPU is True, forcely use CPU to do the calculation rather than GPU.
        saveDir: basestring
            full dirpath to save the outputs.
        kwargs: dict
            dictionary of additional arguments.

    Examples:
        test::blabla

            from scginpy.gin import scgin 
            scgin_obj = scgin.scGIN()
    """
    def __init__(self,
                 project_name : str = "scgin_test",
                 data = None,
                 data_path = None,
                 useCPU: bool = False,
                 show_all_genepairs: bool = False,
                 saveDir : str = os.getcwd(), 
                 *kwargs):
        super().__init__()
        ## project attributes
        self._project_name = project_name
        self._tmpDir = tempfile.mkdtemp()
        self._saveDir = saveDir
        self._dstamp = "".join([str(de) for de in datetime.date.today().timetuple()[:3]])
        ## data attributes
        self._data = data
        self._data_path = data_path
        self._genename_annotation = None
        self._genename_annotation_filepath = None
        self._genepair_annotation = None # genepair indice table with genepair's gene ids represents gene orders displayed on the genename annotation table
        self._genepair_annotation_filepath = None
        ## performed results 
        self._background_spearman_mats = UserList() # SMO
        self._background_spearman_mats_filepath = []
        self._background_spearman_dists = None
        self._background_spearman_dists_filepath = None
        self._target_spearman_mats = UserList() # SMO
        self._target_spearman_mats_filepath = []
        self._deviation_distances = None
        self._deviation_distances_filepath = None
        self._deviation_annotations = None
        self._deviation_annotations_filepath = None
        self._b2t_index_guild_mats = None
        self._b2t_index_guild_mats_filepath = None
        self._show_all_genepairs = show_all_genepairs
        ## memory attributes
        logger.info("check memory conditions")
        if useCPU:
            self._devices = "cpu"
        else:
            self._devices = check_memory_conditions(show_message=False)
        ## set data, genename_annotation and genepairname_annotaion by set_data()
        if (not isnone(self._data)) or (not isnone(self._data_path)):
            if isnone(self._data):
                data = load_file(self._data_path)
                self.set_data(data)
            else:
                logger.info("data already be assigned, input data path is not used.")
                self._data_path = None
                self.set_data(show_all_genepairs=self._show_all_genepairs)
        else:
            logger.warning("scGIN's data has not been assigned, please set it by `set_data()`.")
        os.makedirs(self._saveDir, exist_ok = True)
            

    def set_data(self, data = None, data_path = None, if_output: bool = True, show_all_genepairs: bool = False):
        """set data.
        
        set scGIN's input data of cellxgene counts and scGIN's reference genenames/geneindices, scGIN's reference genepair names/genepair indicies.

        Args:
            data: [anndata.AnnData, pd.DataFrame, np.array]
                scGIN's data that contains the cellxgene counts.
            data_path: basestring
            if_output: boolean
                if save generate genename annotation file (and/or genepair indice table)
            show_all_genepairs: boolean
                generate genepair indice table, this step may take a long time to save genepairs if there is too much genes.

        """
        if not isnone(data):
            if not isnone(data_path):
                logger.warning("data is already assigned, data from "+data_path+" will not be loaded")
            if not isnone(self._data):
                logger.warning("data is already assigned, original data will be overwritten")
        else:
            if not isnone(data_path):
                if not isnone(self._data_path):
                    logger.warning("data filepath will be overwritten")
                if not isnone(self._data):
                    logger.warning("data is already assigned, original data will be overwritten")
                self._data_path = data_path
                data = load_file(data_path)
                self._data = data
            else:
                if not isnone(self._data):
                    data = self._data
                else:
                    print("data is already assigned")

        if isinstance(data, anndata.AnnData):
            self._genename_annotation = pd.DataFrame(data.var.index.tolist(), columns=['gene'])
            self._data = np.array(data.X.todense())
            logger.info("obtained data in anndata AnnData format")
        elif isinstance(data, pd.DataFrame):
            self._genename_annotation = pd.DataFrame(data.columns.tolist(), columns=['gene'])
            self._data = np.array(data)
            logger.info("obtained data in pandas dataframe format")
        elif isinstance(data, np.ndarray):
            logger.warning("Gene names/symbols are not available with np.ndarray format data, using index as the gene name.")
            self._genename_annotation = pd.DataFrame(np.arange(data.shape[1]), columns=['gene'])
            self._data = data
        else:
            logger.error("data format is not identified")
            return None

        # genepairname_annotation = pd.DataFrame(list(combinations(self._genename_annotation.gene,2)), columns=['gene1','gene2'])
        # genepairindice_annotation = pd.DataFrame(list(combinations(np.arange(self._data.shape[1]),2)), columns=['gind1','gind2'])
        # self._genepair_annotation = pd.concat([genepairname_annotation, genepairindice_annotation], axis=1)
        # del data, genepairname_annotation, genepairindice_annotation
        
        if show_all_genepairs:
            self._genepair_annotation = pd.DataFrame(torch.combinations(torch.tensor(np.arange(self._genename_annotation.shape[0])), r=2).numpy(),
                                                    columns = ['gind1', 'gind2'])
            
        if if_output:
            self.save_genename_annotation()
            if show_all_genepairs:
                self.save_genepair_annotation()

        release_memory_occupation()


    def genenames_to_geneindices(self, genenames: list, genename_annotation = None):
        """transform scGIN's gene names to gene indices.

        Args:
            genenames: list[string]
                a list of gene names
            genename_annotation: pandas.DataFrame
                a pandas.DataFrame table with an index column ranked by number and a column named gene with gene names(symbols).

        Returns:
            geneindices: list[integer]

        """
        if isnone(genename_annotation):
            if isnone(self._genename_annotation):
                if isnone(self._genename_annotation_filepath):
                    logger.warning("failed to change genenames to indice, reference genename table and genename filepath are both unavailable.")
                else:
                    genename_annotation = load_file(self._genename_annotation_filepath)
            else:
                genename_annotation = self._genename_annotation
        geneindices = [genename_annotation.reset_index().set_index('gene').loc[genenames][0] for genename in genenames]
        return geneindices

    
    def geneindices_to_genenames(self, geneindicies: list, genename_annotation = None):
        """transform scGIN's gene indices to gene names.

        Args:
            geneindices: list[integer]
                a list of gene indices
            genename_annotation: pandas.DataFrame
                a pandas.DataFrame table with an index column ranked by number and a column named gene with gene names(symbols).

        """
        if isnone(genename_annotation):
            if isnone(self._genename_annotation):
                if isnone(self._genename_annotation_filepath):
                    logger.warning("failed to change genenames to indice, reference genename table and genename filepath are both unavailable.")
                else:
                    genename_annotation = load_file(self._genename_annotation_filepath)
            else:
                genename_annotation = self._genename_annotation
        genenames = [genename_annotation.loc[geneindex][0] for geneindex in geneindicies]
        return genenames


    def show_target_ggpairs_info(self):
        """show target genepair information.
        """
        if isnone(self._target_spearman_mats):
            logger.warning("no target cell groups found.")
            return None
        for smo in self._target_spearman_mats:
            print(smo.get_name())
            smo.show_info()
            print()


    def show_background_ggpairs_info(self):
        """show background genepair information. 
        """
        if isnone(self._background_spearman_mats):
            logger.warning("fitted distributions of genepairs' background spearman correlations have not been calculated")
            return None
        for smo in self._background_spearman_mats:
            print(smo.get_name())
            smo.show_info()
            print()


    def generate_target_GIN(self, 
                            data = None, 
                            cellcond = None, 
                            metatable = None,
                            min_expr: float = -1.0,
                            cellidx = None,
                            geneidx = None,
                            label: str = "_target_",
                            maxgpair: int = 10000,
                            outpath = None,
                            temp_outpath = None,
                            if_use_temporary: bool = False,
                            if_append_to_target: bool = True,
                            if_return: bool = False,
                            if_with_genepairs: bool = True):
        """Generate gene-gene correlation network based on spearman correlation analysis in selected cells.

        Generate target gene-gene spearman correlation network with selected cells.

        Args:
            data: [anndata.Anndata, pd.DataFrame, np.ndarray]
                dataset contains the cellxgene count mat.
            cellcond: dict
                cell conditions applied in the metatable with {features: feature_values}.
            metatable: pd.DataFrame
                metainfo table for single cell data contains features and cells' feature values.
            min_expr: float
                the minimum expression of average gene expression per gene across cells.
            cellidx: [list, np.array]
                the cell indices of cells in data for selections.
            geneidx: [list, np.array]
                the gene indices of genes in data for selections.
            #refgenelist: pandas.DataFrame
            #    default is None. A pd.DataFrame table with an index column contains gene name rank by number and a column named `gene` contains gene names for scGIN's data.
            label: basestring
                default is "_target_". To label outputs from target genepairs' correlation network.
            maxgpair: integer
                default is 10000. Data with genepairs larger than maxgpair will be deviced to splitted iterations by `maxgpair` genepairs. Recommend to set maxgpair value to <10000 while using torch gpus.
            outpath: basestring
                the full output filepath for the generated spearman correlation matrix.
            temp_outpath: basestring
                the temporary output filepath for the splitted spearman correlation matrix.
            if_use_temporary: boolean
                default is False. If if_use_temporary is true, use the temporary output to store the temporary files.
            if_return: boolean
                default is False. If if_return is true, return the cellxgene spearman correlation matrix (Warning, not SMO object).  
            if_append_to_target: boolean
                default is True. If background spearman correlation calls this function, if_append_to_target will be set to False.
            if_with_genepairs: boolean
                default is True. If if_with_genepairs is true, save the list of genepair indices to file.

        """
        ## set timestamp and label
        dstamp = self._dstamp
        # label = label
        savesuffix = ".pickle"
        
        ## check output path and temporary output path
        if isnone(outpath):
            savedir = self._saveDir + os.sep + label ## output to subdirectories: _target_, _background_
        else:
            if not os.path.exists(outpath):
                try:
                    os.makedirs(outpath, exist_ok = True)
                except PermissionError:
                    print("your operations are not granted under the path")
                    raise
                except (RuntimeError, TypeError, NameError, ValueError):
                    print("given path is invalid, please reset your output path")
                    raise  
            savedir = outpath      
        if isnone(temp_outpath):
            tmpdir = self._tmpDir
        else:
            if not os.path.exists(temp_outpath):
                try:
                    os.makedirs(temp_outpath, exist_ok = True)
                except PermissionError:
                    print("your operations are not granted under the path")
                    raise
                except (RuntimeError, TypeError, NameError, ValueError):
                    print("given path is invalid, please reset your temp output path")
                    raise
            tmpdir = temp_outpath

        ## substract required count_matrix from data
        ## transform input data format to np.ndarray format
        if isnone(data):
            if isnone(self._data):
                logger.error("scGIN's data has not been assigned yet, please use function `set_data` to initialize the data or input the data parameter.")
                return None
        else:
            if not isnone(self._data):
                logger.warning("scGIN's data has already assigned, please use function `set_data` to reassigned the data before this step.")
                return None
            self.set_data(data=data)
        data = self.get_data()
        genename_annotation = self.get_genename_annotation()
        # genepairname_annotation = self.get_genepair_annotation()

        ## generate selected cell indices
        cidx = np.arange(data.shape[0]) # cellcond = ['B cell','18m']
        savemeta = self._project_name
        if not isnone(cellcond):
            if isinstance(cellcond,dict):
                if not isinstance(metatable, pd.DataFrame):
                    logger.warning("cellcond (cell conditons) is not used because metatable is not provided!")
                else:
                    cidx = [np.where(np.isin(metatable[key],cellcond[key]))[0] for key in cellcond.keys()]
                    cidx = reduce(lambda x, y: np.intersect1d(x, y), cidx)
                    savemeta = "_".join([savemeta] + [cellcond[key] for key in cellcond.keys()])
            else:
                logger.error("cellcond (cell conditions) should be a dict object.")
        if not isnone(cellidx):
            if isinstance(cellidx,torch.Tensor):
                cellidx = cellidx.numpy()
            cidx = np.intersect1d(cidx, cellidx)
            savemeta = "_".join([savemeta] + ["with_usergiven_clist"])
        logger.info(str(len(cidx))+" cells selected.")
        
        ## generate gene-gene pair indices (ggpairs)
        ggidx = np.arange(data.shape[1])
        if min_expr > 0:
            ggidx = np.where(np.mean(data,axis=0)>min_expr)[0] # ggidx location is using the rank of the genes from the input data as the coordinates.
        if not isnone(geneidx): 
            ggidx = np.intersect1d(ggidx, geneidx)  # geneidx is also using the rank of the genes from the input data as the coordinates. 
        logger.info(str(len(ggidx))+" genes selected.")
        ## obtain required sub-ndarray
        cellxgene = data[cidx,:][:,ggidx]

        ## ggpairs partition
        MAXGPAIR = maxgpair 
        if len(ggidx) > 10000:
            MAXGPAIR = 10000
        if maxgpair > 10000:
            if self._devices == "cpu":
                logger.Warning("Spearman correlation will be calculated per "+str(maxgpair)+" genepairs (>=10000) on cpu. \
                If this step is crashed, please adjust your MAXGPAIR to a smaller number.")
            else:
                logger.Warning("Spearman correlation will be calculated per "+str(maxgpair)+" genepairs (>=10000) on gpus. \
                If this step is crashed, please adjust your MAXGPAIR to a smaller number.")
        
        order = list(combinations(range(len(ggidx)),2)) # this order is using  the input data as the coordinates.
        orders = []
        if len(order)>MAXGPAIR:
            par = len(order) // MAXGPAIR
            for ipar in range(par+1):
                if (ipar+1)*MAXGPAIR > len(order):
                    end = len(order)
                else:
                    end = (ipar+1)*MAXGPAIR
                orders.append(order[ipar*MAXGPAIR:end])
        else:
            orders = [order]
        del order

        ## looped calculation
        devices = self._devices
        cellxgene_spearman_corrmat = []
        npar = 0
        if isinstance(devices,str):
            dno = 1
        else:
            dno = len(devices)
        for ipar in range(len(orders)//dno+1):
            logger.debug(str(ipar)+"th partition started")
            start = dno*ipar
            if len(orders) <= dno*(ipar+1):
                end = len(orders)
                idevice = len(orders) - dno*ipar
            else:
                end = dno*(ipar+1)
                idevice = dno
            for idnum in range(idevice):
                npar += 1 
                if devices != "cpu":
                    idname = str(idnum)+"th "+torch.cuda.get_device_name(idnum) 
                    logger.debug(str(ipar)+"th partition on "+idname+" started")
                    cxg1 = torch.tensor(cellxgene[:,[iord[0] for iord in orders[start+idnum]]],device=torch.device(idnum))
                    cxg2 = torch.tensor(cellxgene[:,[iord[1] for iord in orders[start+idnum]]],device=torch.device(idnum))
                    tmpRes = compute_rank_correlation_2d(cxg1, cxg2).view(-1,1)
                    if if_use_temporary:
                        tlabel = label + str(start + idnum)
                        tmpfile_for_tmpRes = tmpdir + os.sep + savemeta + "." + tlabel + "." + dstamp + savesuffix
                        os.makedirs(os.path.dirname(tmpfile_for_tmpRes), exist_ok = True)
                        with open(tmpfile_for_tmpRes, 'wb') as p:
                            pickle.dump(tmpRes.cpu(), p)
                        logger.debug("save partial results to "+tmpfile_for_tmpRes)
                    else:
                        cellxgene_spearman_corrmat.append(tmpRes.cpu())
                    del cxg1, cxg2, tmpRes
                    release_memory_occupation()
                else:
                    idname = devices
                    logger.debug(str(ipar)+"th partition on "+idname+" started")
                    cxg1 = torch.tensor(cellxgene[:,[iord[0] for iord in orders[start+idnum]]],device=devices)
                    cxg2 = torch.tensor(cellxgene[:,[iord[1] for iord in orders[start+idnum]]],device=devices)
                    tmpRes = compute_rank_correlation_2d(cxg1, cxg2).view(-1,1)
                    if if_use_temporary:
                        tlabel = label + str(start + idnum)
                        tmpfile_for_tmpRes = tmpdir + os.sep + savemeta + "." + tlabel + "." + dstamp + savesuffix
                        with open(tmpfile_for_tmpRes, 'wb') as p:
                            pickle.dump(tmpRes, p)
                        logger.debug("save partial results to "+tmpfile_for_tmpRes)
                    else:
                        cellxgene_spearman_corrmat.append(tmpRes)
                    del cxg1, cxg2, tmpRes
                    release_memory_occupation()
        del cellxgene
        release_memory_occupation()

        ## combine results
        if if_use_temporary:
            partitions = range(npar)
            temprary_files = []
            for ipar in partitions: 
                tlabel = label + str(ipar)
                tmpfile_for_tmpRes = tmpdir + os.sep + savemeta + "." + tlabel + "." + dstamp + savesuffix
                with open(tmpfile_for_tmpRes, "rb") as p:
                    if ipar==0:
                        cellxgene_spearman_corrmat = pickle.load(p)
                    else:
                        cellxgene_spearman_corrmat = torch.cat((cellxgene_spearman_corrmat, pickle.load(p)),dim=0)
                os.remove(tmpfile_for_tmpRes)
        else:
            for ipar in range(len(cellxgene_spearman_corrmat)):
                if ipar==0:
                    tmp_spearman_corrmat = cellxgene_spearman_corrmat[ipar]
                else:
                    tmp_spearman_corrmat = torch.cat((tmp_spearman_corrmat, cellxgene_spearman_corrmat[ipar]),dim=0)
            cellxgene_spearman_corrmat = tmp_spearman_corrmat
            del tmp_spearman_corrmat

        ## generate annotated gene pairs (in pandas.DataFrame, with columns [genename1, genename2])
        # ggpairs = list(combinations([genename[i] for i in ggidx],2))
        # ggpairs= list(combinations(geneindice_list,2)) # this order is using the scGIN's data as the coordinates.

        if if_with_genepairs:
            ggidx0 = np.array(genename_annotation.index) ## ggidx0 is using the gene rank in the scGIN's data as the coordinate system for input data's gene rank.
            geneindice_list = ggidx0[ggidx]
            ggpairs = pd.DataFrame(torch.combinations(torch.tensor(geneindice_list),r=2).numpy(),columns=['gind1','gind2'])

        ## return measured gene-gene correlations and genepair names
        if if_return:
            ## add spearman correlation object to scGIN._target_spearman_mats UserList
            auto_load = False
            smo_name = savemeta
            smo_notes = savemeta + "." + label + "." + dstamp
            if if_with_genepairs and 'ggpairs' in locals():
                smo = SMO(name = smo_name, 
                          data = cellxgene_spearman_corrmat,
                          datatype = "target",
                          annotation =ggpairs,
                          notes = smo_notes, 
                          auto_load=auto_load)
            else:
                smo = SMO(name = smo_name, 
                          data = cellxgene_spearman_corrmat,
                          datatype = "target",
                          notes = smo_notes, 
                          auto_load=auto_load)
            if if_append_to_target:
                self._target_spearman_mats.append(smo) 
            ## return target spearman correlation matrix
            if if_with_genepairs:
                return cellxgene_spearman_corrmat, ggpairs
            else:
                return cellxgene_spearman_corrmat
        else:
            ## save target_spearman_mats to assigned filepath
            ofile = savedir + os.sep + savemeta + "." + label + "." + dstamp + savesuffix
            if if_with_genepairs:
                gfile = savedir + os.sep + savemeta + "." + label + "ggpairs" + "." + dstamp + savesuffix
                save_file(ggpairs, filepath = gfile)
                logger.info("selected cellxgene's genename has been saved in " + gfile)
            save_file(cellxgene_spearman_corrmat, filepath = ofile)
            logger.info("selected cellxgene's spearman correlation matrix has been saved in " + ofile)

            ## add spearman correlation object to scGIN._target_spearman_mats UserList
            auto_load = False
            smo_name = savemeta
            smo_notes = savemeta + "." + label + "." + dstamp
            if not if_with_genepairs:                 
                if 'gfile' in locals(): 
                    smo = SMO(name = smo_name, 
                              datatype = "target", 
                              data_path = ofile, 
                              annotation_path = gfile,
                              notes = smo_notes,
                              auto_load = auto_load)
                else:
                    smo = SMO(name = smo_name, 
                              datatype = "target", 
                              data_path = ofile, 
                              notes = smo_notes,
                              auto_load = auto_load)                    
            else:
                if 'gfile' and 'ggpairs' in locals():
                    smo = SMO(name = smo_name, 
                              data = cellxgene_spearman_corrmat, 
                              datatype = "target", 
                              data_path = ofile, 
                              annotation = ggpairs,
                              annotation_path = gfile,
                              notes = smo_notes,
                              auto_load = auto_load)
                else:
                    smo = SMO(name = smo_name, 
                              data = cellxgene_spearman_corrmat, 
                              datatype = "target", 
                              data_path = ofile, 
                              notes = smo_notes,
                              auto_load = auto_load)
            if if_append_to_target:
                self._target_spearman_mats.append(smo)         

            return None

        
    def generate_background_GIN(self, 
                                data = None, 
                                iter_num: int = 500, 
                                rand_cell_num: int = 1000,
                                cellcond = None,
                                metatable = None,
                                cellidx = None,
                                geneidx = None,
                                label: str = "_background_",
                                if_expressed: bool = True, 
                                min_expr: float = 1, 
                                maxgpair: int = 10000,
                                outpath = None,
                                temp_outpath = None,
                                if_output : bool = True,
                                if_combine_output : bool = True,
                                if_return: bool = False):
        """generate background gene interation networks.

        Using cell x gene count matrix, randomly permutation of gene interaction network. 

        Gene number has a big impact on the calculation time. So filter out those dark genes that don't express across all cells is a recommended step here.
        But this step may impair the observation of dark genes that have abnormal activated genepair's relationships and behaviours under data analysis from pathological conditions.

        Args:
            iter_num : int
                the iteration number for random permutation.
            rand_cell_num : int
                the cell number for sampling.
            cellcond : dict
                cell condition.
            cellidx : np.array
                using offered list of cell indices to filter data's cells.
            geneidx : np.array
                using offered list of gene indices to filter data's genes.
            #refgenelist: list
            #    gene name for the input data before filtration.
            label : str
                default '_background_'. Label to identify output background results by filename.  
            if_expressed: boolean
                default is True. If if_expressed is true, genes with average gene expression less than 0 will be filtered.
            min_expr : float
                average gene expression threshould for filtration of low-expressed and no-expressed genes.
            outpath : basestring
                setting the output filepath for saving the combined gene-gene spearman correlation matrix.            
            temp_outpath : basestring
                setting the output filepath for saving the temporary gene-gene spearman correlation matrix (with incomplete genepairs).
            if_output: boolean.
                default is True. If if_output is true, save background spearman correlation matrix.
            if_combine_output: boolean.
                default is True. If if_combine_output is true, delete temporary splitted intermediate files and combine the splitted to a merged file.
            if_return: boolean.
                default is False. if if_return is true, return the randomly sampled gene-gene spearman correlation matrix with each row represents a genepair and each column represents a random sampling.
        
        """
        if_use_temporary = True ## save the temperary sampled results

        if not if_return:
            if_output = True
        
        ## generate selected gene idx and metatable
        if not isnone(self._data):
            if not isnone(data):
                logger.warning("scGIN's data already initialized, using scGIN's data instead of inputted data")
        else:
            if isnone(data):
                logger.error("scGIN's data has not been assigned yet, please use function `set_data` to initialize the data or input the data parameter.")
                raise
            else:
                self.set_data(data=data)
        genename_annotation = self._genename_annotation

        ## generate selected cell indices
        cidx = np.arange(self._data.shape[0]) # cellcond = ['B cell','18m']
        if not isnone(cellcond):
            if isinstance(cellcond,dict):
                if not isinstance(metatable, pd.DataFrame):
                    logger.error("cellcond (cell conditons) is not used because metatable is not provided!")
                else:
                    cidx = [np.where(np.isin(metatable[key],cellcond[key]))[0] for key in cellcond.keys()]
                    cidx = reduce(lambda x, y: np.intersect1d(x, y), cidx)
            else:
                logger.error("cellcond (cell conditions) should be a dict object.")
                return None
        if not isnone(cellidx):
            if isinstance(cellidx,torch.Tensor):
                cellidx = cellidx.numpy()
            cidx = np.intersect1d(cidx, cellidx)
        logger.info(str(len(cidx))+" cells selected.")

        ## generate permutation matrix
        # all_mat = tf.random.uniform([iter_num, rand_cell_num], 
        #                             maxval=self._data.shape[0], 
        #                             dtype=tf.int32)
        if len(cidx) < rand_cell_num:
            rand_cell_num = int(0.3*len(cidx))
        all_mat = tf.random.uniform([iter_num, rand_cell_num],
                                    maxval = len(cidx),
                                    dtype = tf.int32)
        all_mat = cidx[all_mat.numpy()]
        all_mat = torch.tensor(all_mat, dtype=torch.int32)

        savesuffix = self._dstamp + ".pickle"
        # label = label
        if isnone(outpath):
            savedir = self._saveDir + os.sep + label
        else:
            savedir = outpath
        os.makedirs(savedir, exist_ok = True)
        if isnone(temp_outpath):
            tmpdir = self._tmpDir
        else:
            tmpdir = temp_outpath
        os.makedirs(tmpdir, exist_ok = True)

        if isnone(geneidx):
            geneidx = np.arange(self._data.shape[1])
        if if_expressed:
            gidx = np.where(np.mean(self._data,axis=0)>min_expr)[0]
            geneidx = np.intersect1d(geneidx, gidx)

        if not if_use_temporary:
            logger.warning("if_use_temporary is off, if memory is not enough the program will crash")
         
        ## tensor loop
        for i in range(iter_num):
            slabel = label + str(i+1) + "th_"
            ## call generate_target_GIN
            if not if_use_temporary:
                if i == 0:            
                    background_spearman_corrmat, ggpairs = self.generate_target_GIN(cellidx = all_mat[i,:],
                                                                                    geneidx = geneidx,
                                                                                    maxgpair = maxgpair,
                                                                                    metatable = metatable,
                                                                                    min_expr = min_expr,
                                                                                    outpath = None,
                                                                                    label = slabel,
                                                                                    temp_outpath = tmpdir,
                                                                                    if_use_temporary = if_use_temporary,
                                                                                    if_with_genepairs = True,
                                                                                    if_append_to_target = False,
                                                                                    if_return = if_return)
                else:
                    tmpRes = self.generate_target_GIN(cellidx = all_mat[i,:],
                                                      geneidx = geneidx,
                                                      maxgpair = maxgpair,
                                                      metatable = metatable,
                                                      min_expr = min_expr,
                                                      outpath = None,
                                                      label = slabel,
                                                      temp_outpath = tmpdir,
                                                      if_use_temporary = if_use_temporary,
                                                      if_with_genepairs = False,
                                                      if_append_to_target = False,
                                                      if_return = if_return)
                background_spearman_corrmat = torch.cat((background_spearman_corrmat,tmpRes), dim=1)
            else:
                if i == 0:
                    self.generate_target_GIN(cellidx = all_mat[i,:],
                                             geneidx = geneidx,
                                             maxgpair = maxgpair,
                                             metatable = metatable,
                                             min_expr = min_expr,
                                             outpath = savedir,
                                             label = slabel,
                                             temp_outpath = tmpdir,
                                             if_use_temporary = if_use_temporary,
                                             if_return = if_return,
                                             if_append_to_target = False,
                                             if_with_genepairs = True)
                    gfile = savedir + os.sep + self._project_name + "_with_usergiven_clist" + "." + slabel + "ggpairs" + "." + savesuffix
                    ggpairs = load_file(gfile)
                else:
                    self.generate_target_GIN(cellidx = all_mat[i,:],
                                             geneidx = geneidx,
                                             maxgpair = maxgpair,
                                             metatable = metatable,
                                             min_expr = min_expr,
                                             outpath = savedir,
                                             label = slabel,
                                             temp_outpath = tmpdir,
                                             if_use_temporary = if_use_temporary,
                                             if_return = if_return,
                                             if_append_to_target = False,
                                             if_with_genepairs = False)

        ## initiate smo pre-parameters
        smo_name = self._project_name + "_random_resample_" + str(iter_num) + "x" + str(rand_cell_num)
        smo_notes = self._project_name + "_with_usergiven_clist." + label + self._dstamp + "\n" \
                    + " with iteraction number: " + str(iter_num) + ", with resampled cell number per iteration: " + str(rand_cell_num)
        smo_dtype = "background"
        auto_load = False
        ## output and transfer results to self._background_spearman_corrmats
        if not if_use_temporary:
            if if_output:
                background_spearman_corrmat_filepath = savedir + os.sep + self._project_name + "_with_usergiven_clist." + label + "." + savesuffix
                with open(background_spearman_corrmat_filepath, 'wb') as p:
                    pickle.dump(background_spearman_corrmat, p)
                self._background_spearman_mats_filepath.append(background_spearman_corrmat_filepath) 
                logger.info("background spearman correlation matrix has been combined into "+background_spearman_corrmat_filepath)
                background_ggpairs_filepath = savedir + os.sep + self._project_name + "_with_usergiven_clist." + label + "1th_ggpairs." + savesuffix
                ## add smo to scGIN._background_spearman_corrmat
                smo = SMO(name = smo_name, 
                          data = background_spearman_corrmat, 
                          datatype = smo_dtype, 
                          data_path = background_spearman_corrmat_filepath, 
                          annotation = ggpairs,
                          annotation_path = background_ggpairs_filepath,
                          notes = smo_notes,
                          auto_load = auto_load)
                self._bakcground_spearman_mats.append(smo)
            if if_return:
                ## add smo to scGIN._background_spearman_corrmat
                smo = SMO(name = smo_name,
                          data = background_spearman_corrmat, 
                          datatype = "background",
                          annotation = ggpairs,
                          notes = smo_notes,
                          auto_load = auto_load)
                self._background_spearman_mats.append(smo)
                ## return
                return background_spearman_corrmat, ggpairs
        else:
            ## combine all random selected ggpair spearman correlations
            # load_background_GIN(temp_outpath)
            for i in range(iter_num):
                slabel = label + str(i+1) + "th_"
                temp_pfile = savedir + os.sep + self._project_name + "_with_usergiven_clist." + slabel + "." + savesuffix
                with open(temp_pfile, 'rb') as p:
                    if i == 0:
                        background_spearman_corrmat = pickle.load(p)
                    else:
                        background_spearman_corrmat = torch.cat((background_spearman_corrmat, pickle.load(p)), dim=1)
                if if_combine_output:
                    os.remove(temp_pfile)
            ## output
            if if_combine_output:
                background_spearman_corrmat_filepath = savedir + os.sep + self._project_name + "_with_usergiven_clist." + label + "." + savesuffix
                with open(background_spearman_corrmat_filepath, 'wb') as p:
                    pickle.dump(background_spearman_corrmat, p)
                logger.info("background spearman correlation matrix has been combined into "+background_spearman_corrmat_filepath)
                self._background_spearman_mats_filepath.append(background_spearman_corrmat_filepath) 
                background_ggpairs_filepath = savedir + os.sep + self._project_name + "_with_usergiven_clist." + label + "1th_ggpairs." + savesuffix                
                ## transfer smo to scgin._background_spearman_corrmats
                smo = SMO(name = smo_name,
                          data_path = background_spearman_corrmat_filepath,
                          datatype = "background",
                          annotation = ggpairs,
                          annotation_path = background_ggpairs_filepath,
                          notes = smo_notes,
                          auto_load = auto_load
                        )
                self._background_spearman_mats.append(smo)
            if if_return:
                ## transfer smo to scgin._background_spearman_corrmats
                smo = SMO(name = smo_name,
                          data = background_spearman_corrmat,
                          datatype = "background",
                          annotation = ggpairs,
                          notes = smo_notes,
                          auto_load = auto_load
                        )
                self._background_spearman_mats.append(smo)
                ## return
                return background_spearman_corrmat


    def generate_background_spearman_distributions(self, clear_cached: bool = True, if_return: bool = False, if_output: bool = True):
        """generate background spearman distributions.

        After generate background spearman correlation matrix, using normal distribution to fit each genepair's spearman correlation's distribution curve and return the distribution parameters.

        Args:
            clear_cached: boolean
                default is True. If clear_cached is true, release memory space of scGIN stored background spearman correlation matrix.
            if_return: boolean
                default is False. If if_return is true, return the generated normal distribution parameters for the genepairs.

        """
        background_spearman_distributions = UserList()
        for smo in self._background_spearman_mats:
            background_spearman_corrmat = smo.get_data()
            background_spearman_distributions.append(np.vstack([background_spearman_corrmat.mean(axis=1), background_spearman_corrmat.std(axis=1)]))
        self._background_spearman_dists = background_spearman_distributions
        if clear_cached:
            bid = 0
            savedir = self._saveDir
            savesuffix = self._dstamp + ".pickle"
            for smo in self._background_spearman_mats:
                bid += 1
                slabel = "_background_" + str(bid) + "th"
                filepaths = smo.show_filepaths(if_return = True)
                if isnone(filepaths):
                    background_spearman_corrmat_filepath = savedir + os.sep + self._project_name + "_with_usergiven_clist." + slabel + "." + savesuffix
                    logger.error(self._project_name + "'s background spearman matrice are not saved.")
                    smo.save_data(dtype = "data", 
                                  filepath = background_spearman_corrmat_filepath, 
                                  dformat = ".pickle")
                smo.clear_data(if_clear_filepath = False, 
                               if_clear_annotation = True, 
                               if_clear_notes = False) # using default clear_data
        if if_output:
            self.export_background_spearman_distributions()
        if if_return:
            return background_spearman_distributions


    def get_data(self, if_load = False):
        """get scGIN's data.
        """
        if isnone(self._data):
            if isnone(self._data_path):
                logger.error("scGIN's data and data filepath are not available, please reload the data with `set_data`")
                return None
            if if_load:
                self._data = load_file(self._data_path)
                return self._data
            return load_file(self._data_path)
        else:
            return self._data


    def get_genename_annotation(self, if_load = False):
        """get scGIN data's genename annotation.
        """
        if isnone(self._genename_annotation):
            if isnone(self._genename_annotation_filepath):
                logger.error("genename annotation and annotation filepath are not available, please reload the data's genename annotation with `set_data`")
                return None
            if if_load:
                self._genename_annotation = load_file(self._genename_annotation_filepath)
                return self._genename_annotation
            return load_file(self._genename_annotation_filepath)
        else:
            return self._genename_annotation


    def get_genepair_annotation(self, if_load=False):
        """get scGIN data's genepair annotation.
        """
        if isnone(self._genepair_annotation):
            if isnone(self._genepair_annotation_filepath):
                logger.error("genepair annotation is not available, please reload the data with `set_data`")
                return None
            return load_file(self._genenpair_annotation_filepath)
        else:
            return self._genepair_annotation


    def get_background_spearman_distributions(self):
        """get background spearman distributions

        Check whether scGIN has background spearman correlation distributions. If not, calculate the fitted normal distributions for the background spearman correlation and return the parameters for these distributions.
        
        """
        if isnone(self._background_spearman_dists):
            background_spearman_distributions = self.generate_background_spearman_distributions(if_return = True)
        else:
            background_spearman_distributions = self._background_spearman_dists
        return background_spearman_distributions
    

    def export_background_spearman_distributions(self, filepath = None, label = "_background_"):
        """export normal distributions for the background spearman correlations.

        Args:
            filepath: basestring
                default is None, output parameters of background spearman distributions to the assigned filepath.

        """
        ## save data to pickle
        background_spearman_distributions = self.get_background_spearman_distributions()
        if not isnone(background_spearman_distributions):
            if not isnone(filepath):
                if os.path.exists(filepath):
                    logger.error(filepath + " is already existed, the original file will be overwritten.")
                self._background_spearman_distributions_path = filepath
                save_file(background_spearman_distributions, filepath = self._background_spearman_distributions_path)
            else:
                savedir = self._saveDir + os.sep + label
                savesuffix = self._dstamp + ".pickle"
                slabel = "_background_distributions_params_"
                filepath = savedir + os.sep + self._project_name + "_with_usergiven_clist." + slabel + "." + savesuffix
                self._background_spearman_distributions_path = filepath
                save_file(background_spearman_distributions, filepath = self._background_spearman_distributions_path)
        else:
            logger.error("result for background spearman distributions is not found, please check the process")


    def generate_deviation_distance(self,
                                    background_spearman_distributions = None,
                                    target_spearman_corrmats = None,
                                    guild_idx_mat = None,
                                    background_annotation_list = None,
                                    target_annotation_list = None):
        """generation of deviation distance.

        generation of deviation distance between background spearman correlation mats and target spearman correaltion matrices.

        Args:
            background_spearman_distributions: List[np.array] 
                list of 2D np.array. 0th row represents miu of normal distribution and 1th row represents sigma of normal distribution. Each column represents a genepair.
            target_spearman_corrmats : List[np.array, torch.tensor]
                list of 2D np.array. Each element represents a target spearman correlation matrix.
            guild_idx_mat : List[np.array]
                guild indice matrix which is constituted by pairs of target matrice and background matrice. Each element-row represents a candidate genepair and element-column represents the genepair indice from the background and the genepair indice from the target.
            background_annotation_list : list
                default is None. The list of annotations for every background conditions.
            target_annotation_list : list or None
                default is None. The list of annotations for every target conditions.

        """
        if isnone(background_spearman_distributions):
            logger.info("number of " + self._project_name + "'s background spearman corrmats: " + str(background_spearman_distributions))
        if isnone(target_spearman_corrmats):
            logger.error(self._project_name + "'s target spearm corrmat has not been assigned yet.")
        if isnone(guild_idx_mat):
            bshapes = [b.shape[0] for b in background_spearman_distributions]
            tshapes = [t.shape[0] for t in target_spearman_corrmats]
            if len(np.unique(bshapes + tshapes)) != 1:
                logger.error(self._project_name + "'s target ggpairs' numbers are different from background ggpairs', please provide guild_idx_mat or use the same ggpairs.")
            else:
                guild_idx_mat = [np.array([np.arange(bshapes[0])]*2).swapaxes(0,1)]*(len(bshapes)*len(tshapes))
        deviation_distances = UserList()
        if (not isnone(background_annotation_list)) and (not isnone(target_annotation_list)):
            add_annotation = True
            deviation_annotations = UserList()
        bn = 0         
        for background_spearman_distribution in background_spearman_distributions:
            tn = 0
            for target_spearman_corrmat in target_spearman_corrmats:
                ggpairs_indice = guild_idx_mat[bn*(len(target_spearman_corrmats))+tn]
                
                deviation_distance = []

                for i in range(ggpairs_indice.shape[0]):
                    ib = ggpairs_indice[i,0] ## background ggpairs available index
                    it = ggpairs_indice[i,1] ## target ggpairs available index

                    deviation_distance.append(2*(torch.distributions.Normal(loc=background_spearman_distribution[0,ib],
                                                                            scale=background_spearman_distribution[1,ib]).cdf(target_spearman_corrmat[it][0]) - 0.5))
                deviation_distance = torch.stack(deviation_distance)
                deviation_distances.append(deviation_distance)
                
                # deviation_distance=list(map(lambda i: 2*(stats.distributions.norm.cdf(target_spearman_corrmat[i], 
                #                                     loc=background_spearman_distribution[0,i], 
                #                                     scale=background_spearman_distribution[1,i])-0.5)[0], 
                #                                     range(background_spearman_distribution.shape[1])))
                # deviation_distances.append(deviation_distance)
                if 'deviation_annotations' in locals():
                    deviation_annotation = background_annotation_list[bn] + "\t" + target_annotation_list[tn]
                    deviation_annotations.append(deviation_annotation)
                tn += 1
            bn += 1
        if add_annotation:
            return deviation_distances, deviation_annotations
        else:
            return deviation_distances
       

    def generate_deviation_dist(self, if_return: bool = False, if_export = True):
        """ generate deviation distance matrix.

        Args:
            add_annotation: boolean
                default is False. If if_return is true, return deviation distances.

        """
        # background_annotation_list = None
        # target_annotation_list = None

        if isnone(self._background_spearman_dists):
            background_spearman_distributions = self.generate_background_spearman_distributions()
        else:
            background_spearman_distributions = self._background_spearman_dists
        logger.info("number of " + self._project_name + "'s background spearman corrmats: " + str(len(background_spearman_distributions)))
        background_annotation_list = []
        background_ggpairs_list = []
        for smo in self._background_spearman_mats:
            background_annotation_list.append(smo.get_name())
            background_ggpairs_list.append(smo.get_annotation())

        if isnone(self._target_spearman_mats):
            logger.error(self._name + "'s target spearm corrmat has not been assigned yet.")
        else:
            logger.info("number of " + self._project_name + "'s target spearman corrmats: " + str(len(self._target_spearman_mats)))
            target_annotation_list = []
            target_spearman_corrmats = UserList()
            target_ggpairs_list = []
            for smo in self._target_spearman_mats:
                target_spearman_corrmats.append(smo.get_data())
                target_annotation_list.append(smo.get_name())
                target_ggpairs_list.append(smo.get_annotation())
    
        ## generate guild index matrice
        bshapes = [b.shape[1] for b in background_spearman_distributions]
        tshapes = [t.shape[0] for t in target_spearman_corrmats]
        if len(np.unique(bshapes + tshapes)) == 1:
            guild_idx_mat = [np.array([np.arange(bshapes[0])]*2).swapaxes(0,1)]*(len(bshapes)*len(tshapes))
        else:
            guild_idx_mat = []
            # print("backgrounds: "+str(len(background_ggpairs_list))+" x targets: "+str(len(target_ggpairs_list)))
            for ib in range(len(background_ggpairs_list)):
                for it in range(len(target_ggpairs_list)):
                    # print("backgrounds "+str(ib)+": "+str(len(background_ggpairs_list[ib])))
                    # print("targets "+str(it)+": "+str(len(target_ggpairs_list[it])))
                    bxt_ggpairs = check_genepairs(ggpairs1 = background_ggpairs_list[ib], ggpairs2 = target_ggpairs_list[it])
                    guild_idx_mat.append(bxt_ggpairs)
                    del bxt_ggpairs

        if (not isnone(target_annotation_list)) and (not isnone(background_annotation_list)):
            deviation_distances, deviation_annotations = self.generate_deviation_distance(background_spearman_distributions = background_spearman_distributions,
                                                                                            target_spearman_corrmats = target_spearman_corrmats,
                                                                                            guild_idx_mat = guild_idx_mat,
                                                                                            background_annotation_list = background_annotation_list,
                                                                                            target_annotation_list = target_annotation_list)
            self._deviation_distances = deviation_distances
            self._deviation_annotations = deviation_annotations
            self._b2t_index_guild_mats = guild_idx_mat
            if if_return:
                return deviation_distances, deviation_annotations, guild_idx_mat
            else:
                self.export_deviation_results()
        else:
            deviation_distances = self.generate_deviation_distance(background_spearman_distributions = background_spearman_distributions,
                                                                     target_spearman_corrmats = target_spearman_corrmats,
                                                                     guild_idx_mat = guild_idx_mat)
            self._deviation_distances = deviation_distances
            self._b2t_index_guild_mats = guild_idx_mat
            if if_return:
                return deviation_distances, guild_idx_mat
            else:
                self.export_deviation_results()


    # @TODO
    # def get_deviation_results(self):
    #     """get deviation results.
    #     """
    #     if isnone(self._deviation_distances):
    #         logger.warning("result for deviation distances is not found")
    #         if not isnone(self._deviation_distances_filepath):
    #             deviation_distances = load_file(filepath = self._deviation_distances_filepath)
    #     if isnone(self._deviation_annotations):
    #         logger.warning("deviation distance group annotations are not found")
    #         if not isnone(self._deviation_annotations_filepath):
    #             deviation_annotations = load_file(filepath = self._deviation_annotations_filepath)
    #     if isnone(self._b2t_index_guild_mats):
    #         logger.warning("guild genepair indices for background cell and target cell groups are not found")
    #         if not isnone(self._b2t_index_guild_mats_filepath):
    #             b2t_index_guild_mats = load_file(filepath = self._b2t_index_guild_mats_filepath)
    #     deviation_objs = np.intersect1d(['deviation_distances', 'deviation_annotations', 'b2t_index_guild_mats'],[key for key in locals().keys()])
        


    def export_deviation_results(self, label = "_deviation_dists_", savedir = None):
        """save deviation results to assigned directory.
        """
        if isnone(savedir):
            savedir = self._saveDir + os.sep + label
        savesuffix = self._dstamp + ".pickle"

        if not isnone(self._deviation_distances):
            if isnone(self._deviation_distances_filepath):
                label = "_deviation_distances_"
                self._deviation_distances_filepath = savedir + os.sep + self._project_name + label + "." + savesuffix
            else:
                logger.warning("deviation distance file already exists, will be overwritten with new data")
            save_file(self._deviation_distances, filepath = self._deviation_distances_filepath)
        else:
            logger.warning("result for deviation distances is not found.")

        if not isnone(self._b2t_index_guild_mats):
            if isnone(self._b2t_index_guild_mats_filepath):
                label = "_deviation_genepair_guild_indices_"
                self._b2t_index_guild_mats_filepath = savedir + os.sep + self._project_name + label + "." + savesuffix
            else:
                logger.warning("file of genepairs indices for background and target cell groups already exists, will be overwritten with new data")
            save_file(self._b2t_index_guild_mats, self._b2t_index_guild_mats_filepath)
        else:
            logger.warning("result for genepair indices of background and target cell groups is not found.")

        if not isnone(self._deviation_annotations):
            if isnone(self._deviation_annotations_filepath):
                label = "_deviation_annotations_"
                self._deviation_annotations_filepath = savedir + os.sep + self._project_name + label + "." + savesuffix
            else:
                logger.warning("deviation annotation file already exists, will be overwritten with new data")
            save_file(self._deviation_annotations, self._deviation_annotations_filepath)
        else:
            logger.warning("result for deviation annotations is not found")

    
    def get_save_direcotry(self):
        """get scGIN's output dirpath.
        """
        print(self._saveDir)
        return self._saveDir
   

    def get_project_name(self):
        """get scGIN's name.
        """
        print(self._project_name)
        return self._project_name


    def get_devices(self, show_message=False):
        """get memory devices of the machine.

        Args:
            show_message: boolean
                default is False. If show_message is true, display the devices' current conditions and return the devices.
        """
        if show_message:
            get_devices_conditions()
        return self._devices


    def get_devices_conditions(self, devices = None):
        """get devices' conditions.

        Args:
            devices: List
                list of the devices' names for checking memory conditions.
    
        """
        if isnone(devices):
            check_memory_conditions(devices=self._devices, show_message=True, return_devices=False)
        else:
            if devices == self._devices or np.isin(devices, self._devices):
                check_memory_conditions(devices=devices, show_message=True, return_devices=False)
            else:
                print("Warning: " + devices + " is not applied for scgin.")
                check_memory_conditions(devices=devices, show_message=True, return_devices=False)
        return None


    # def get_background_spearman_corrmats(self):
    #     if self._background_spearman_mats != None:
    #         return self._background_spearman_mats
    #     else:
    #         if self._background_spearman_mats_filepath != None:
    #             logger.warning(self._project_name + "'s background spearman mats in memory already been cleared, reload from "+self._background_spearman_mats_filepath)
    #             return load_file(self._background_spearman_mats_filepath)
    #         else:
    #             logger.warning(self._project_name + "'s background spearman mats have not been generated yet.")


    # def get_target_spearman_corrmats(self):
    #     if self._target_spearman_mats != None:
    #         return self._target_spearman_mats
    #     else:
    #         if self._target_spearman_mats_filepath != None:
    #             logger.warning(self._project_name + "'s target spearman mats in memory already been clered, reload from "+self._target_spearman_mats_filepath)
    #             return load_file(self._background_spearman_mats_filepath)
    #     return None


    def set_save_directory(self, saveDir: str):
        """set scGIN's output directory.
        """
        print(saveDir)
        self._saveDir = saveDir
        return None


    def set_project_name(self, project_name: str):
        """set scGIN's project name.
        """
        print(project_name)
        self._project_name = project_name
        return None


    def set_devices(self, devices):
        """set scGIN's devices.
        """
        if devices == "cpu":
            self._devices = devices
        elif isinstance(devices, torch.device):
            self._devices = [devices]
        elif isinstance(devices, list):
            if isinstance(devices[0], torch.device):
                self._devices = devices
            else:
                print("Only cpu and gpus [torch.device] are allowed here.")
                raise
        else:
            print("Only cpu and gpus [torch.device] are allowed here.")
            raise
        return None


    def save_genepair_annotation(self, filepath = None, dformat = ".csv", **kwargs):
        """save genepair annotation.
        """
        if isnone(self._genepair_annotation):
            logger.error("data genepair annotation has not been assigned yet, please set it with `set_data`")
            return None
        avail_dformats = [".pickle", ".csv", ".tsv", ".xlsx"]
        if dformat not in avail_dformats:
            logger.error("data format is not identified, please use "+", ".join(avail_dformats))
            return None
        if isnone(filepath):
            dirpath = self._saveDir
            filename = self._project_name + "._all_ggpair_indices." + self._dstamp + dformat
            filepath = dirpath + os.sep + filename
            self._genepair_annotation_filepath = filepath
        save_file(data = self._genepair_annotation, data_format = dformat, filepath = filepath)


    def save_genename_annotation(self, filepath = None, dformat = ".csv", **kwargs):
        """save genename annotation.
        """
        if isnone(self._genename_annotation):
            logger.error("data genename annotation has not been assigned yet, please set it with `set_data`")
            return None
        avail_dformats = [".pickle", ".csv", ".tsv", ".xlsx"]
        if dformat not in avail_dformats:
            logger.error("data format is not identified, please use "+", ".join(avail_dformats))
            return None
        if isnone(filepath):
            dirpath = self._saveDir
            filename = self._project_name + "._all_genenames." + self._dstamp + dformat
            filepath = dirpath + os.sep + filename
            self._genename_annotation_filepath = filepath
        save_file(data = self._genename_annotation, data_format = dformat, filepath = filepath)


    def save_data(self, filepath = None, **kwargs):
        """save scGIN's data.
        """
        if isnone(filepath):
            dirpath = self._saveDir
            filename = self._project_name + "._scGIN_." + self._dstamp + ".pickle" 
            filepath = dirpath + os.sep + filename
        save_file(data = self, filepath = filepath, data_format = ".pickle", **kwargs)
  