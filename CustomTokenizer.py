import string
import re
import Utilities as Utils
import logging


### Basic tokenizer, that splits on whitespace and handles punctuation
def tokenize_line(line):
    line_text = line.strip() # remove trailing whitespaces
    all_punct_pattern = '[' + string.punctuation + ']'
    punct_tokeep_pattern = "[-'<>]" # do not space out hyphen, apostrophe, and <>
    punct_pattern = re.sub(punct_tokeep_pattern, '', all_punct_pattern)

    line_spacedpuncts = re.sub(pattern="(" + punct_pattern + ")", repl=r" \1 ", string=line_text)
    line_final = re.sub(pattern="  ", repl=" ", string=line_spacedpuncts)

    line_tokens = line_final.split()
    return line_tokens

### Use the tokenizer to space out the punctuation and a the tokens in a text file, and rewrite it
def spaceout_tokens_in_text_file(fpath):
    logging.info("Spacing out tokens (including punctuation) for the file: " + fpath)
    out_lines = []
    with open(fpath, 'r') as in_file:
        for line in in_file:
            line_tokens = tokenize_line(line)
            out_line = ' '.join(line_tokens)
            out_lines.append(out_line)

    out_text = '\n'.join(out_lines)
    with open(fpath, 'w') as out_file:
        out_file.write(out_text)


### Memorize the frequencies of the words in the corpus
def count_file(text_filepaths_ls):
    logging.info("Counting vocabulary frequencies in the training file...")
    freq_dictionary = {}

    for in_filepath in text_filepaths_ls:
        with open(in_filepath, 'r') as in_file:
            for line in in_file:
                line_tokens = tokenize_line(line)
                for token in line_tokens:
                    try:
                        freq = freq_dictionary[token]
                        freq_dictionary[token] = freq + 1
                    except KeyError:
                        freq_dictionary[token] = 1
    return freq_dictionary


### After tokenizing a corpus, we should replace any token with frequency < min_freq with <unk>
def insert_unk_in_text(text_fpath, freq_dictionary, min_frequency):
    logging.info("Replacing words with frequency <"  + str(min_frequency) + " in the file:" + text_fpath)
    out_lines = []
    tot_tokens_counter = 0
    unk_tokens_counter = 0

    with open(text_fpath, 'r') as in_file:
        for line in in_file:
            line_tokens = line.split()
            line_new_tokens = []
            for token in line_tokens:
                if freq_dictionary[token] >= min_frequency:
                    line_new_tokens.append(token)
                else:
                    line_new_tokens.append(Utils.UNK_TOKEN)
                    unk_tokens_counter = unk_tokens_counter + 1
                tot_tokens_counter = tot_tokens_counter + 1
            out_line = ' '.join(line_new_tokens)
            out_lines.append(out_line)

    logging.info("Total number of tokens = ")
    logging.info("Number of " + Utils.UNK_TOKEN + "tokens = ")

    out_text = '\n'.join(out_lines)
    with open(text_fpath, 'w') as out_file:
        out_file.write(out_text)
