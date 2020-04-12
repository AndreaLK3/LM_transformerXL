import Utilities as Utils
import os
import subprocess
import re
from InputFacilities import CustomTokenizer as CT
import numpy as np

### Step 1
### create_text_from_wikidump(path_to_wiki_dump, destination_foldername), turns a .bz2 compressed archive
### into multiple folders (e.g. AA, AB, etc.) with files wiki_00, wiki_01, etc.. Each files has multiple <doc>s
def create_text_from_wikidump(path_to_wiki_dump, dataset_dirpath):
    path_to_wikiextractor = os.path.join('wikiextractor', 'WikiExtractor.py')
    path_to_dest_folder = os.path.join(dataset_dirpath, Utils.SOURCES_FOLDER, 'plain_wiki')
    cmd = ['python2',
           path_to_wikiextractor,
           '--no_templates',
           '--min_text_length',
           '50',
           '--filter_disambig_pages',
           '-o',
           path_to_dest_folder,
           path_to_wiki_dump]
    process = subprocess.Popen(cmd)
    process.wait()


### Step 2: As done in line_remover.py in corpus_builder:
### - Remove trailing whitespaces, if carriage return and linefeed are the only remaining elements
### - Ignore the <doc> and </doc> lines, since the title of an article is below <doc>
def adjust_plain_wikifiles(allinput_dirpath, output_dirpath):
    subdirs = list(filter(lambda foldername: not (foldername.startswith('.')), os.listdir(allinput_dirpath)))
    for subdir in subdirs:
        plain_wikifiles_names = os.listdir(os.path.join(allinput_dirpath, subdir))
        plain_wikifiles_fpaths = [os.path.join(allinput_dirpath, subdir, fname) for fname in plain_wikifiles_names]
        for plain_wiki_fpath in plain_wikifiles_fpaths:
            refine_wikitext(plain_wiki_fpath, subdir, output_dirpath)


def refine_wikitext(plain_wiki_fpath, input_subdirectory, output_dirpath):
    if not(os.path.isdir(output_dirpath)): os.mkdir(output_dirpath)
    with open(plain_wiki_fpath, 'r') as plain_wikifile:
        plain_path, basename = os.path.split(plain_wiki_fpath)
        output_fpath = os.path.join(output_dirpath, input_subdirectory + '_clean_' + basename)
        with open(output_fpath, 'w') as outfile:
            for line in plain_wikifile:
                if line.startswith('<doc') or line.startswith('</doc>'):
                    continue # skipping
                if re.match(r'^\s*$', line):  # ignore empty lines (eg. with only carriage return and linefeed)
                    continue # \s matches any whitespace character; equivalent to the set [ \t\n\r\f\v]
                outfile.write(line)


### Step 3: Reunite all the files in clean_wiki into training-validation-test (80-10-10), so we can use iterators
def reunite_corpus_splits(clean_wiki_dirpath, output_dirpath, fraction_included_dataset):

    clean_wiki_fpaths = sorted([os.path.join(clean_wiki_dirpath, fname) for fname in os.listdir(clean_wiki_dirpath)])
    tot_files = len(clean_wiki_fpaths)
    out_train_file = open(os.path.join(output_dirpath, 'train.txt'),'w')
    out_valid_file = open(os.path.join(output_dirpath, 'valid.txt'),'w')
    out_test_file = open(os.path.join(output_dirpath, 'test.txt'),'w')

    training_indices = np.random.choice(range(tot_files), size=int(0.8*tot_files*fraction_included_dataset),
                                                 replace=False, p=None)
    valid_and_test_subfiles_indices = np.random.choice(set(range(tot_files)).difference(set(training_indices)),
                                                 size=int(0.2 * tot_files * fraction_included_dataset),
                                                 replace=False, p=None)
    validation_indices = valid_and_test_subfiles_indices[0: len(valid_and_test_subfiles_indices) // 2]
    test_indices = valid_and_test_subfiles_indices[len(valid_and_test_subfiles_indices) // 2:]

    for i in range(tot_files):
        with open(clean_wiki_fpaths[i], 'r') as in_subfile:
            in_subfile_text = in_subfile.read()
            if i in training_indices:
                out_train_file.write(in_subfile_text)
            elif i in validation_indices:
                out_valid_file.write(in_subfile_text)
            elif i in test_indices:
                out_test_file.write(in_subfile_text)

    out_train_file.close()
    out_valid_file.close()
    out_test_file.close()


### Step 4: Postprocessing (spacing out punctuation, tokenizing, and inserting <unk> tokens in the place of rare words.
def postprocess_corpus(dataset_dirpath, min_frequency):

    train_fpath = os.path.join(dataset_dirpath, 'train.txt')
    valid_fpath = os.path.join(dataset_dirpath, 'valid.txt')
    test_fpath = os.path.join(dataset_dirpath, 'test.txt')
    in_fpaths = [train_fpath, valid_fpath, test_fpath]

    # Tokenize:
    for in_fpath in in_fpaths:
        CT.spaceout_tokens_in_text_file(in_fpath)

    # the dictionary of frequencies is built on the training set
    frequency_dictionary = CT.count_file(in_fpaths)

    # Replace rare words with <unk> tokens
    for in_fpath in in_fpaths:
        CT.insert_unk_in_text(in_fpath, frequency_dictionary, min_frequency)


