# Objective:
# specifying a dataset_dirpath,
#   either:  - using the module PrepareCorpus,
#              extract the train, valid and test.txt files from the xml wikipedia dump
#            - train a transformer-xl model
#   or: - load the saved transformer-xl model

import PrepareCorpus as PC
import os
import Utilities as U
import logging
import Filesystem as F
import importlib
import torch
import sys

##### Filesystem checks and utilities
def check_dataset_splits_present(dataset_dirpath):
    return ( os.path.exists(os.path.join(dataset_dirpath, 'train.txt'))
            and os.path.exists(os.path.join(dataset_dirpath, 'valid.txt'))
            and os.path.exists(os.path.join(dataset_dirpath, 'test.txt')) )

def create_folders_ifneeded(folderpaths_ls):
    for folderpath in folderpaths_ls:
        if not(os.path.isdir(folderpath)):
            os.makedirs(folderpath)
#####


### Generic function to gather the text from the wiki dump of a given language
def make_dataset_splits(wikidump_fpath, dataset_dirpath):
    U.init_logging('Input-make_dataset_splits.log')
    create_folders_ifneeded([dataset_dirpath])

    if not check_dataset_splits_present(dataset_dirpath):
        logging.info("Gathering text from the WikiDump...")

        PC.create_text_from_wikidump(wikidump_fpath, dataset_dirpath)

        plaintext_dirpath = os.path.join(dataset_dirpath, 'plain_wiki')
        clean_wiki_dirpath = os.path.join(dataset_dirpath, 'clean_wiki')
        PC.adjust_plain_wikifiles(plaintext_dirpath, clean_wiki_dirpath)

        PC.reunite_corpus_splits(clean_wiki_dirpath, dataset_dirpath)


def get_model_txl(dataset_dirpath):
    model_dirpath = os.path.join(dataset_dirpath, F.MODEL_SUBFOLDER)
    create_folders_ifneeded([model_dirpath])
    model_fpath = os.path.join(model_dirpath, 'model.pt')

    if os.path.exists(model_fpath):

        # adjusting the sys.path to allow us to load a model that is not in transformer-xl/pytorch
        sys.path.append(os.path.join(os.getcwd(), 'transformer-xl', 'pytorch'))
        sys.path.append(os.path.join(os.getcwd(), 'transformer-xl', 'pytorch', 'utils'))
        

        os.chdir(os.path.join('transformer-xl', 'pytorch'))
        data_utils = importlib.import_module('data_utils')
        txl_model = torch.load('model.pt')

        os.chdir(os.path.join('..', '..'))