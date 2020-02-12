import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
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


######## Drawing graphs #########

def display_ygraph_fromfile(npy_fpath, axis_labels=None):
  data_y_array = np.load(npy_fpath, allow_pickle=True)
  plt.plot(data_y_array)
  plt.xticks(range(0, len(data_y_array), 1))
  plt.yticks(range(0, int(max(data_y_array)) + 1, 1))
  plt.xlim((0, len(data_y_array)))
  plt.ylim((0, max(data_y_array)))
  plt.grid(b=True, color='lightgrey', linestyle='-', linewidth=0.5)
  if axis_labels is not None:
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])


# For now, intended to be use with training_losses and validation_losses
def display_xygraph_from_files(npy_fpaths_ls):
  overall_max = 0
  legend_labels = ['Training loss', 'Validation loss']
  for i in range(len(npy_fpaths_ls)):
    npy_fpath = npy_fpaths_ls[i]
    xy_lts_array = np.load(npy_fpath, allow_pickle=True)
    plt.plot(xy_lts_array.transpose()[0], xy_lts_array.transpose()[1], label=legend_labels[i])
    array_max = max(xy_lts_array.transpose()[1])
    overall_max = array_max if array_max > overall_max else overall_max
  plt.ylim((0, overall_max))
  ax = plt.axes()
  ax.legend()


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