import Utilities as Utils
import logging
import os
import subprocess
import re


### Step 1
### create_text_from_wikidump(path_to_wiki_dump, destination_foldername), turns a .bz2 compressed archive
### into multiple folders (e.g. AA, AB, etc.) with files wiki_00, wiki_01, etc.. Each files has multiple <doc>s
def create_text_from_wikidump(path_to_wiki_dump, dataset_dirpath):
    path_to_wikiextractor = os.path.join('wikiextractor', 'WikiExtractor.py')
    path_to_dest_folder = os.path.join(dataset_dirpath,'plain_wiki')
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
def reunite_corpus_splits(clean_wiki_dirpath, output_dirpath):

    clean_wiki_fpaths = sorted([os.path.join(clean_wiki_dirpath, fname) for fname in os.listdir(clean_wiki_dirpath)])
    tot_files = len(clean_wiki_fpaths)
    out_train_file = open(os.path.join(output_dirpath, 'train.txt'),'w')
    out_valid_file = open(os.path.join(output_dirpath, 'valid.txt'),'w')
    out_test_file = open(os.path.join(output_dirpath, 'test.txt'),'w')

    for i in range(tot_files):
        with open(clean_wiki_fpaths[i] , 'r') as in_subfile:
            in_subfile_text = in_subfile.read()
            if i < 0.2 * tot_files:# i < (0.8 * tot_files):
                out_train_file.write(in_subfile_text)
            elif (0.2 * tot_files) < i < (0.25 * tot_files): # (0.8 * tot_files) < i < (0.9 * tot_files):
                out_valid_file.write(in_subfile_text)
            elif (0.25 * tot_files) < i < (0.3 * tot_files): # i > 0.9*tot_files
                out_test_file.write(in_subfile_text)

    out_train_file.write(' ' + Utils.UNK_TOKEN + ' ') # we add it if the corpus does not have it.
    out_train_file.close()
    out_valid_file.write(' ' + Utils.UNK_TOKEN + ' ')
    out_valid_file.close()
    out_test_file.write(' ' + Utils.UNK_TOKEN + ' ')
    out_test_file.close()
