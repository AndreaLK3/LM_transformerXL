import re
import string

import Utilities as Utils

### From a given text (i.e. for a given language) we need to create a vocabulary file, where
### each line contains one instance of a word encountered in the training corpus
def create_vocabfile_for_tokenizer(training_corpus_fpath, vocabfile_fpath):
    with open(training_corpus_fpath, 'r') as in_file:
        with open(vocabfile_fpath, 'w') as out_file:
            out_file.write(Utils.UNK_TOKEN + '\n') # manually adding <unk> if it is not present in the corpus
            for line in in_file:
                line_tokens = tokenize_line(line)
                for token in line_tokens:
                    out_file.write(token + '\n')


### Basic tokenizer, that splits on whitespace and handles punctuation
### Used to contruct the language's tokenizer
def tokenize_line(line):
    line_text = line.strip() # remove trailing whitespaces
    all_punct_pattern = '[' + string.punctuation + ']'
    punct_tokeep_pattern = "[-']" # do not remove hyphen and apostrophe
    punct_pattern = re.sub(punct_tokeep_pattern, '', all_punct_pattern)

    line_text_nopunct = re.sub(punct_pattern, ' ', line_text)
    line_tokens = line_text_nopunct.split()
    return line_tokens