import datetime
import os
from io import open

import numpy as np
import pandas as pd
import matplotlib
# Force matplotlib to not use any Xwindows backend, to avoid errors when executing on remote from command line
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Utilities import NUM_DISPLAYED
import Utilities as Utils


##### Text output #####

##### Cleaning up special characters from the output

def adjust_puncts(line):
  # simple rules of detokenization

  line = line.replace(' @-@ ', '-')
  line = line.replace(' @,@ ', ',')
  line = line.replace(' @.@ ', '.')
  line = line.replace(' . ', '. ')
  line = line.replace(' , ', ', ')
  line = line.replace(' : ', ': ')
  line = line.replace(' ; ', '; ')
  line = line.replace(" 's ", "'s ")
  line = line.replace(' ( ', ' (')
  line = line.replace(' ) ', ') ')
  return line

def format_text(tokens):
  line = ''
  for token in tokens:
    if token == '<eos>':
      line += '\n'
    else:
      line += token
      line += ' '
  line = adjust_puncts(line)
  return line

def write_sequence_output(context, start_idx, reference, generation, out_path):
  with open(out_path, 'w', encoding='utf-8') as f:
    f.write('Start line: {}'.format(start_idx) + '\n')
    f.write('Context len: {}'.format(len(context)) + '\n')
    f.write('\n' * 2 + '-' * 80 + '\ncontext=\n')
    f.write(format_text(context))
    f.write('\n' + '-' * 80 + '\n')
    f.write('\n' * 2 + '-' * 80 + '\nGeneration=\n')
    f.write(format_text(generation))
    f.write('\n' * 2 + '-' * 80 + '\nreference[:args.max_gen_len]=\n')
    f.write(format_text(reference))


def write_nextwords_incsv(context, nextwords, probs_tensor):
  #If the directory s not there yet, create it. If it contains > 50 files, clean it
  dirname = Utils.DIR_WORDPROBABILITIES
  if not os.path.exists(dirname):
      os.makedirs(dirname)

  filename = create_filename("", context, ".csv")

  with open(os.path.join(Utils.DIR_WORDPROBABILITIES, filename), mode='w') as suggestions_file:

    all_probs_ls = [round(element.item(),4) for element in probs_tensor]
    word_prob_df = pd.DataFrame(list(zip(nextwords, all_probs_ls)))
    word_prob_df.to_csv(suggestions_file, sep=',', mode='w', header=False, index=False)

######


###### Building graphs of predicted words and probabilities #####

# Build a pyplot graph that show the probabilities for choosing the next word, as produced by the Transformer-XL LM
def create_graphs(context_tokens, words, probs):
  saves_directory = Utils.DIR_WORDPROBABILITIES

  context_adjusted = [ token if token!='<eos>' else "\n" for token in context_tokens[-128:] ]
  context_printable = " ".join(context_adjusted)
  context_to_display = "Preceding context: \n\n" + "\"" + context_printable

  # the histogram of the data
  fig,ax = plt.subplots(figsize=(12, 8), dpi=128)
  x = np.arange(len(probs[0:NUM_DISPLAYED]))
  plt.bar(x, probs[0:NUM_DISPLAYED])
  plt.xticks(x, words[0:NUM_DISPLAYED])
  plt.yticks([i * 0.05 for i in range(20)])

  props = dict(boxstyle='round4', facecolor='lightgrey', alpha=0.4)

  # place a text box in upper left in axes coords
  t = plt.text(0.3, 0.98, context_to_display, transform=ax.transAxes, fontsize=8,
           horizontalalignment='left',
           bbox=props, verticalalignment='top', wrap=True) #bbox=props,


  #plt.show()
  fname = create_filename("", context_printable, '.png')

  plt.savefig(os.path.join(saves_directory, fname))

  #plt.close('all')


def create_filename(part_name, context, extension):
  d = datetime.datetime.now()
  day_hour_minute = "_".join([d.strftime("%j"), d.strftime("%H"), d.strftime("%M")])
  filename = day_hour_minute + '_' +part_name + '_' +str(context[-12:]) + '_' + extension
  return filename

########


######## Drawing graphs of losses #########

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
