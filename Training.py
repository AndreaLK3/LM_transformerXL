import importlib
import subprocess
import Input
from enum import Enum
import os
import sys
import Utilities as Utils

# A Dataset refers to the tuple (dataset_dirpath, wikidump_fname)
class Dataset(Enum):
    DANISH_WIKI = (os.path.join(Utils.DATASETS_FOLDER, Utils.DANISH_WIKI),
                   'dawiki-latest-pages-articles.xml.bz2')
    WIKITEXT_103 = (os.path.join(os.getcwd(), Utils.DATASETS_FOLDER, Utils.WIKITEXT_103),
                    None)



def train(dataset):

    sys.path.append(os.path.join(os.getcwd(), 'transformer-xl', 'pytorch'))

    Input.make_dataset_splits(dataset)

    os.chdir(os.path.join('transformer-xl', 'pytorch'))
    # the parameters are modified from: the default ones + run_wt103_base.sh
    bash_command = ['python3',
                    'train.py',
                    '--cuda',
                    '--data',
                    dataset[0], # the dataset_dirpath. The original default was: ../data/wikitext-103/
                    '--dataset',
                    'wt103', # since it can only be a choice of: choices=['wt103', 'lm1b', 'enwik8', 'text8']
                    '--adaptive',
                    '--n_layer',
                    '16', # taken from run_wt103_base.sh. Can be modified if needed (e.g. to save memory)
                    '--d_model',
                    '410',
                    '--n_head',
                    '10',
                    '--d_head',
                    '41',
                    '--d_inner',
                    '2100',
                    '--dropout',
                    '0.1',
                    '--dropatt',
                    '0.0',
                    '--optim',
                    'adam',
                    '--lr',
                    '0.00025',
                    '--warmup_step',
                    '0',
                    '--max_step',
                    '200000',
                    '--tgt_len',
                    '150',
                    '--mem_len',
                    '150',
                    '--eval_tgt_len',
                    '150',
                    '--batch_size',
                    '60',
                    '--multi_gpu',
                    '--gpu0_bsz',
                    '4',
                    '--eval-interval', # changed by me from 4000
                    '2000'
                    ]

    subprocess.call(bash_command)
    os.chdir(os.path.join('..','..'))