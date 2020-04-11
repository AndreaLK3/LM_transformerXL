import importlib
import subprocess
import Input
import torch
import os
import sys
import Utilities as Utils

###### In order to train a Transformer-XL Language Model on a given dataset/language, for example:
###### import Input
###### dataset= Input.Dataset.DANISH
###### import Training
###### Training.train(dataset)

def train(dataset, min_frequency_forvocab=5, inputdataset_fraction=1):

    sys.path.append(os.path.join(os.getcwd(), 'transformer-xl', 'pytorch'))

    Input.make_dataset_splits(dataset, min_frequency_forvocab, inputdataset_fraction)

    os.chdir(os.path.join('transformer-xl', 'pytorch'))
    # the parameters are modified from: the default ones + run_wt103_base.sh
    bash_command = ['python3',
                    'train.py',
                    '--cuda',
                    '--data',
                    os.path.join('..','..',dataset.value[0]), # the dataset_dirpath. The original default was: ../data/wikitext-103/
                    '--dataset',
                    'wt103', # since it can only be one of: choices=['wt103', 'lm1b', 'enwik8', 'text8']
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
                    '32', # changed from 60, to try to avoid 'RunTimeError: CUDA out of memory'
                    '--multi_gpu',
                    '--gpu0_bsz',
                    '4',
                    '--eval-interval', # changed by me from 4000
                    '2000'
                    ]

    subprocess.call(bash_command)
    os.chdir(os.path.join('..','..'))


def evaluate(text_folder):
    sys.path.append(os.path.join(os.getcwd(), 'transformer-xl', 'pytorch'))

    os.chdir(os.path.join('transformer-xl', 'pytorch'))

    bash_command = ['python3',
                    'eval.py',
                    '--cuda',
                    '--data',
                    os.path.join('..','..',text_folder), # where we find test.txt,
                    '--dataset',
                    'wt103',
                    '--tgt_len',
                    '64',
                    '--mem_len',
                    '640',
                    '--clamp_len',
                    '400',
                    '--same_length',
                    '--split',
                    'test']
    subprocess.call(bash_command)
    os.chdir(os.path.join('..', '..'))