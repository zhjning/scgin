# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd 
import anndata
import logging

from scginpy.utils import *

logger = logging.getLogger('smo')


## scGIN._background_spearman_mats and scGIN._target_spearman_mats tuples
class SMO():
    """spearman correlation object.

    Args:
        name: basestring
            name of the spearman correlation object.
        data: [np.array, pd.DataFrame]
            spearman correlation matrix.
        datatype: basestring
            two data types of SMO objects: 1.background; 2.target.
        data_path: basestring
            full data filepath for the spearman correlation matrix of genepairs.
        annotation: tuple list
            tuple list of genepairs' names or indices.            
        annotation_path: basestring
            full annotation filepath for the spearman correlation matrix of genepairs.
        notes: basestring
            a complete description about the spearman correlation object. Default value is None.

    """
    def __init__(self, 
                name: str, 
                data=None, 
                datatype=None, 
                data_path=None, 
                annotation=None,
                annotation_path=None,
                notes=None,
                auto_load: bool=True):
        super().__init__()
        self._name = name
        self._data = data
        self._datatype = datatype
        self._data_path = data_path
        self._annotation = annotation # annotation like ggpairs
        self._annotation_path = annotation_path
        self._notes = notes
        self._auto_load = auto_load
        if not isnone(self._data_path):
            if self._auto_load:
                self.load_data(data_path=self._data_path, annotation_path=self._annotation_path)
        logger.info("finished initialization.")


    def clear_data(self, if_clear_filepath: bool = False, if_clear_annotation: bool = True, if_clear_notes: bool = False):
        """ clear SMO object.

        clear spearman correlation object's spearman correlation matrix (and annotation, notes).

        Args:
            if_clear_filepath: boolean
                default False. If if_clear_filepath is True, clear SMO's data_path and annotaion_path.
            if_clear_annotation: boolean
                default True. If if_clear_annotation is True, clear SMO's annotation.
            if_clear_notes: boolean
                default False. If if_clear_notes is True, clear SMO's notes.

        """
        if isnone(self._data):
            logger.info("cached " + self._name + "'s data is already cleared.")
        else:
            del self._data
            release_memory_occupation()
            self._data = None
            if if_clear_filepath:
                self._data_path = None
                logger.info(self._name + "'s data filepath is reset to None.")
            logger.info("cached " + self._name + "'s data is cleared.")
        if if_clear_annotation:
            if isnone(self._annotation):
                logger.info("cached " + self._name + "'s annotations are already cleared.")
            else:
                del self._annotation
                release_memory_occupation()
                self._annotation = None
                if if_clear_filepath:
                    self._annotation_path = None
                    logger.info(self._name + "'s annotation filepath is reset to None.")
                logger.info("cached " + self._name + "'s annotations are cleared.")
        else:
            logger.warning("annotations are kept. if you want to refresh annotations, please set if_clear_annotation `True`")
        if if_clear_notes:
            self._notes = None
            logger.info(self._name + "'s description is cleared.")


    def reload_data(self, data_path = None, annotation_path = None, if_clear_annotation: bool = True) -> None:
        """reload data.

        reload spearman correlation matrix or annotation of genepairs or both to the memory.

        Args:
            data_path: basestring
                full data filepath for the spearman correlation matrix of genepairs.
            annotation_path: basestring
                full annotation filepath for the spearman correlation matrix of genepairs.
            if_clear_annotation: boolean
                default value is True. If value is true, clear annotation in the memory.

        """
        if isnone(data_path):
            data_path = self._data_path
        if isnone(annotation_path):
            annotation_path = self._annotation_path
        logger.info("if_clear_annotation is set to " + str(if_clear_annotation))
        ## clear SMO's data (and annotation)
        self.clear_data(if_clear_annotation = if_clear_annotation)

        if if_clear_annotation:
            if isnone(annotation_path):
                self.load_data(data_path = data_path)
                logger.debug(self._name + "'s annotation_path is not provided.")
                logger.info(self._name + "'s data is reloaded.")
            else:
                self.load_data(data_path = data_path, annotation_path = annotation_path)
                logger.info(self._name + "'s data and annotation are reloaded.")
        else:
            self.load_data(data_path = data_path)
            logger.info(self._name + "'s data is reloaded.")


    def load_data(self, data_path = None, annotation_path = None):
        """load SMO's spearman correlation matrix or annotation for genepairs.

        Args:
            data_path: basestring
                default is None. SMO's data filepath, if not provided, will apply SMO's default data filepath. 
            annotation_path: basestring
                default is None. SMO's annotation filepath, if not provided, will apply SMO's default annotation filepath.

        """
        if isnone(data_path) and (not isnone(annotation_path)):
            self.load_annotation(annotation_path = annotation_path)
        else:
            if not isnone(self._data):
                logger.info(self._name + "'s data is already loaded. If you want to refresh the data, please use reload_data().")
            else:
                if isnone(self._data_path) and isnone(data_path):
                    logger.warning("failed to load "+ self._name + ". The filepath is not assigned.")
                else:
                    if not isnone(data_path):
                        if not isnone(self._data_path):
                            logger.warning("old filepath :"+self._data_path+" has been refreshed by:\n"+data_path)
                        self._data_path = data_path
                    if not os.path.exists(self._data_path):
                        logger.warning("failed to load "+ self._name + ". The filepath not exists: "+self._data_path)
                    else:
                        self._data = load_file(self._data_path) ## import load_file from scginpy.utils
            ## if annotation_path is provided
            if not isnone(self._annotation):
                logger.info(self._name + "'s annotation are already loaded. If you want to refresh annotations, please use reload_data().")
            else:
                if (not isnone(self._annotation_path)) or (not isnone(annotation_path)):
                    if not isnone(annotation_path):
                        if not isnone(self._annotation_path):
                            logger.warning("old filepath :"+self._data_path+" has been refreshed by:\n"+data_path)
                        self._annotation_path = annotation_path
                    self._annotation = load_file(self._annotation_path) ## import load_file from scginpy.utils


    def load_annotation(self, annotation_path: str):
        """load annotation.

        load annotation with input annotation filepath.

        Args:
            annotation_path: basestring
                full annotation path of the list of SMO's genepairs' names or indices.

        """
        if os.path.exists(annotation_path):
            self._annotation_path = annotation_path
            self._annotation = load_file(annotation_path)
        else:
            logger.warning(annotation_path + " not exists, please check your annotation_path.")


    def add_notes(self, notes: str):
        """load notes.

        add a complete description about the SMO.

        Args:
            notes: basestring
                a complete description about the SMO.

        """
        if not isnone(self._notes):
            logger.debug("previous notes for " + self._name + ": " + self._notes)
            logger.info("reset notes for" + self._name)
        self._notes =notes
        

    def get_name(self):
        """return SMO's name.
        """
        return self._name


    def get_data(self):
        """return SMO's spearman correlation matrix.
        """
        if isnone(self._data):
            self.load_data()
        return self._data


    def get_annotation(self):
        """return the list of SMO's genepairs' names or indices.
        """
        if isnone(self._annotation):
            self.load_annotation(annotation_path = self._annotation_path)
        return self._annotation


    def get_notes(self):
        """return the complete description of SMO.
        """
        return self._notes


    def save_data(self, dtype: str = 'all', filepath = None, dformat: str = '.pickle', **kwargs):
        """save SMO object or SMO's data/annotation.

        Args:
            dtype: basestring
                data type of the to be saved object ('all', 'data', 'annotation').
            filepath: basestring
                default is None. the full filepath of SMO object or SMO's data/annotation.
            dformat: basestring
                default is '.pickle', the saved file format. 
            kwargs: dict()
                other parameters for utils.save_file().

        """
        if isnone(filepath):
            dirpath = self._saveDir
            filename = "smo_" + self._name + dformat 
            filepath = dirpath + os.sep + filename
        if dtype == "all":
            save_file(data = self, filepath = filepath, data_format = dformat, **kwargs)
        elif dtype == "data":
            save_file(data = self._data, filepath = filepath, data_format = dformat, **kwargs)
            if not isnone(self._data_path):
                logger.warning("old data path exists: "+self._data_path+", replaced by "+filepath)
            self._data_path = filepath
        elif dtype == "annotation":
            save_file(data = self._annotation, filepath = filepath, data_format = dformat, **kwargs)
            if not isnone(self._annotation_path):
                logger.warning("old annotation path exists: "+self._annotation_path+", replaced by "+filepath)
            self._annotation_path = filepath
        else:
            logger.error(dtype + " is not a valid dtype, please use 'data' or 'annotation' as the dtype.")

        return(self, filepath)


    def show_filepaths(self, if_return: bool = False):
        """show filepaths of SMO's data and annotation stored.

        Args:
            if_return: boolean
                if if_return is True, return the list of the filepaths to the terminal.
        """
        filepaths = []
        if not isnone(self._data_path):
            print("data's filepath: " + self._data_path)
            filepaths.append(self._data_path)
        if not isnone(self._annotation_path):
            print("annotation's filepath: " + self._annotation_path)
            filepaths.append(self._annotation_path)
        if if_return:
            if len(filepaths) > 0:
                return filepaths
            else:
                return None


    def show_info(self):
        """show information of SMO.
        """
        if not isnone(self._data):
            if isinstance(self._data, list):
                print("size of data: "+str(len(self._data)))
            if isinstance(self._data, torch.Tensor) or isinstance(self._data, np.array) or isinstance(self._data, pd.DataFrame) or isinstance(self._data, anndata.AnnData):
                print("size of data: "+str(self._data.shape))
        if not isnone(self._annotation):
            print("size of annotations: "+str(len(self._annotation)))
        if not isnone(self._notes):
            print("description: " + self._notes)