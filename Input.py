import PrepareCorpus as PC
import os
import Utilities as U
import logging
import importlib
import torch
import sys
from enum import Enum
import Utilities as Utils

# A Dataset refers to the tuple (dataset_dirpath, wikidump_fname)
class Dataset(Enum):
    DANISH_WIKI = (os.path.join(Utils.DATASETS_FOLDER, Utils.DANISH_WIKI),
                   'dawiki-latest-pages-articles.xml.bz2')
    WIKITEXT_103 = (os.path.join(os.getcwd(), Utils.DATASETS_FOLDER, Utils.WIKITEXT_103),
                    None)


##### Filesystem check
def check_dataset_splits_present(dataset_dirpath):
    return ( os.path.exists(os.path.join(dataset_dirpath, 'train.txt'))
            and os.path.exists(os.path.join(dataset_dirpath, 'valid.txt'))
            and os.path.exists(os.path.join(dataset_dirpath, 'test.txt')) )
#####



### Generic function to gather the text from the wiki dump of a given language
def make_dataset_splits(dataset, words_min_frequency=5):
    dataset_dirpath, wikidump_fname = dataset.value

    if not check_dataset_splits_present(dataset_dirpath):
        logging.info("Gathering text from the WikiDump...")

        wikidump_fpath = os.path.join(dataset_dirpath, wikidump_fname)
        PC.create_text_from_wikidump(wikidump_fpath, dataset_dirpath)

        plaintext_dirpath = os.path.join(dataset_dirpath, 'plain_wiki')
        clean_wiki_dirpath = os.path.join(dataset_dirpath, 'clean_wiki')
        PC.adjust_plain_wikifiles(plaintext_dirpath, clean_wiki_dirpath)

        PC.reunite_corpus_splits(clean_wiki_dirpath, dataset_dirpath)
        logging.info("Dataset splits created at: " + dataset_dirpath)

        PC.postprocess_corpus(dataset_dirpath, words_min_frequency)
        logging.info("Corpus files tokenized and post-processed")

    else:
        logging.info("Dataset splits train.txt, valid.txt, test.txt already present at : " + dataset_dirpath +
                     " for the dataset from: " + str(wikidump_fname))

#####


##### When reading from a text for inference
def get_input(flag_text_or_manual, inp_text="", inp_filepath='.'):
  lines=[]
  text=""
  if flag_text_or_manual:
    with open(os.path.join(inp_filepath, 'input_text.txt'), 'r', encoding='utf-8') as f:
      lines = f.readlines()
      text = "\n".join(lines)
  if flag_text_or_manual == False:
    lines = inp_text.split('\n')
    text = inp_text
  return lines, text