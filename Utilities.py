import logging
import sys
import string
import re
import subprocess
import os
import torch

######## Constants #######
NUM_DISPLAYED = 10
UNK_TOKEN = '<unk>'
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

####### Filesystem #######
DATASETS_FOLDER = 'Datasets'

WIKITEXT_103 = 'WikiText-103'
DANISH_WIKI = 'Danish_Wiki'

DIR_WORDPROBABILITIES = 'candidate next words'


####### Logging ########

def init_logging(logfilename, loglevel=logging.INFO):
  for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
  logging.basicConfig(level=loglevel, filename=logfilename, filemode="w",
                      format='%(asctime)s -%(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
  # print(logging.getLogger())
  if len(logging.getLogger().handlers) < 2:
      outlog_h = logging.StreamHandler(sys.stdout)
      outlog_h.setLevel(loglevel)
      logging.getLogger().addHandler(outlog_h)


# Time measurements
def log_chronometer(time_measurements):
  logging.info("Time analysis:")
  for i in range(len(time_measurements) - 1):
    t1 = time_measurements[i]
    t2 = time_measurements[i + 1]
    logging.info('t' + str(i + 1) + ' - t' + str(i) + ' = ' + str(round(t2 - t1, 5)))
########

def create_folders_ifneeded(folderpaths_ls):
    for folderpath in folderpaths_ls:
        if not(os.path.isdir(folderpath)):
            os.makedirs(folderpath)

##### Check GPU memory usage
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
      [
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
      ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


