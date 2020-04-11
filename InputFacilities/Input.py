from InputFacilities import PrepareCorpus as PC
import os
import logging
from enum import Enum
import Utilities as Utils

# A Dataset refers to the tuple (dataset_dirpath, wikidump_fname)
class Dataset(Enum):
    DANISH = (os.path.join(Utils.DATASETS_FOLDER, Utils.DANISH),
                   'dawiki-latest-pages-articles.xml.bz2')
    SPANISH = (os.path.join(Utils.DATASETS_FOLDER, Utils.SPANISH),
                   'eswiki-latest-pages-articles.xml.bz2')
    WIKITEXT_103 = (os.path.join(os.getcwd(), Utils.DATASETS_FOLDER, Utils.WIKITEXT_103),
                    None)


##### Filesystem check
def check_dataset_splits_present(dataset_dirpath):
    return ( os.path.exists(os.path.join(dataset_dirpath, 'train.txt'))
            and os.path.exists(os.path.join(dataset_dirpath, 'valid.txt'))
            and os.path.exists(os.path.join(dataset_dirpath, 'test.txt')) )
#####

### Function to read any .txt files in the Dataset's Sources subfolder, and append them to the dataset splits
def add_nonwiki_sources(dataset_dirpath, fraction_included):
    txt_sources_fnames = list(filter(lambda fname: fname.endswith('.txt'),
                                     os.listdir(os.path.join(dataset_dirpath, Utils.SOURCES_FOLDER))))
    txt_sources_fpath = list(map(lambda source_fname: os.path.join(dataset_dirpath, Utils.SOURCES_FOLDER, source_fname),
                                 txt_sources_fnames))

    out_train_file = open(os.path.join(dataset_dirpath, 'train.txt'), 'a')
    out_valid_file = open(os.path.join(dataset_dirpath, 'valid.txt'), 'a')
    out_test_file = open(os.path.join(dataset_dirpath, 'test.txt'), 'a')

    for source_fpath in txt_sources_fpath:
        with open(source_fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            num_source_lines_for_train = int(0.9*len(lines) * fraction_included)  # changed to 90% 5% 5% split
            num_source_lines_for_valid = int(0.05*len(lines) * fraction_included)
            text_to_add_train = "\n".join(lines[0:num_source_lines_for_train])
            text_to_add_valid = "\n".join(lines[num_source_lines_for_train:num_source_lines_for_train + num_source_lines_for_valid])
            text_to_add_test = "\n".join(lines[num_source_lines_for_train + num_source_lines_for_valid:num_source_lines_for_train + 2*num_source_lines_for_valid])
            out_train_file.write(text_to_add_train)
            out_valid_file.write(text_to_add_valid)
            out_test_file.write(text_to_add_test)

    out_train_file.close()
    out_valid_file.close()
    out_test_file.close()

    return txt_sources_fnames


    

### Generic function to gather the text from the wiki dump of a given language
def make_dataset_splits(dataset, min_freq_forvocab=5, fraction_included=1):
    dataset_dirpath, wikidump_fname = dataset.value

    if not check_dataset_splits_present(dataset_dirpath):
        logging.info("Gathering text from the WikiDump...")

        wikidump_fpath = os.path.join(dataset_dirpath, Utils.SOURCES_FOLDER, wikidump_fname)
        PC.create_text_from_wikidump(wikidump_fpath, dataset_dirpath)

        plaintext_dirpath = os.path.join(dataset_dirpath, Utils.SOURCES_FOLDER, 'plain_wiki')
        clean_wiki_dirpath = os.path.join(dataset_dirpath,  Utils.SOURCES_FOLDER, 'clean_wiki')
        PC.adjust_plain_wikifiles(plaintext_dirpath, clean_wiki_dirpath)

        PC.reunite_corpus_splits(clean_wiki_dirpath, dataset_dirpath, fraction_included)
        logging.info("Dataset splits created at: " + dataset_dirpath)

        txt_sources_fnames = add_nonwiki_sources(dataset_dirpath, fraction_included)
        logging.info("Appended additional sources: " + str(txt_sources_fnames))

        PC.postprocess_corpus(dataset_dirpath, min_freq_forvocab)
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