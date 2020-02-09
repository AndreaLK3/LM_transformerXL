# Analysis of the use case of Writing Assistant:
#
# When writing a text, we have to give from 3 to 10 alternatives for the next word.
# The previous words in the context should help.
#
# More in detail:
# - If we are in an out-of-word setting, namely:
#     * when we are starting to write
#     * when we have just typed a blank space, or a punctuation sign
#     * when we have chosen a proposed alternative (n: choosing a word adds automatically a blank space afterwards)
# THEN our aim is to present 3 to 10 suggestions for the next word.
#
# - If we are in an in-word setting, namely:
#     * when we have just typed a letter, or a number
#     * when we have clicked between characters, either inside a word or exacly at the end of a word
# THEN our objective is to provide a number of alternative completions for that word
# (e.g. we are about to write 'Plant' and we type 'Pla'; the program should propose completions,
# for instance 'Players', 'Plans', 'Place', 'Play', 'Plant')
#
#
# Method:
# 1) Examine the input text (now found in the file 'input_text.txt'. Later on, to be passed by the application)
# 2) Does this input text terminate with a blank space? Alternatively, is it entirely empty?
#   2a) If Yes, then we are in an out-of-word setting:
#       * Apply standard word prediction using Transformer-XL. Obtain a probability distribution over K candidates.
#   2b) If No (i.e. we are terminating with a character or a number), then we are in an in-word setting:
#       * Run through the list of candidates and their probabilities, and gather the first K candidates that start
#         with the specified prefix. Propose these candidates.
#
import string
import torch
import logging
import regex
import re
import Prediction.RecordResults as RR
import Utilities as Utils
import Input as InputFacilities
from Prediction import Prediction_Core as Core
import transformers as tf
import os

CTX_LEN = 256 # number of tokens in a full transformer window
LOGFILE = 'Transformer_LM.log'
LOGLEVEL = logging.WARN

########## Auxiliary function: from the whole text, extract the context for the prediction
def select_context(all_text):
    context = ""
    if len(all_text)==0 : return ""

    pattern_anything_but_pipe = re.compile('([^|])+')
    m = pattern_anything_but_pipe.match(all_text)
    if m is None:
        context = all_text
    else:
        context = m.group(0)
    if len(context)==len(all_text):
        context = context + " " # we want the next word after the end of the text, not the completion of the last

    return context
##########

########## Auxiliary function: given the context, split it into lines and tokenize it
def process_context(context):
    lines = context.strip().split('\n')
    lines = list(filter(lambda l: not (l.isspace()) and len(l) > 0, lines))  # filter out empty lines
    logging.debug(lines)

    # NOTE: if the context was empty, it is necessary to insert a token to avoid an error in the Transformer
    if len(lines) == 0:
        lines = ['<unk>']

    #temp = sum([l.strip().split() + ['<eos>'] for l in lines], [])
    tokens_splitonwhitespace = ' '.join(sum([l.strip().split() + ['<eos>'] for l in lines], []))
    punct_tosplit = (string.punctuation + "’").replace('-', '').replace('<','').replace('>','')
    tokens_spacearoundpunct = (re.sub(pattern='(['+punct_tosplit+'])', repl=r' \1 ', string=tokens_splitonwhitespace)).split()

    context_tokens = tokens_spacearoundpunct[-CTX_LEN:-1]  # keep the last window for the context, ignore the last line's <eos>
    logging.info("Number of tokens in the context: " + str(len(context_tokens)))

    return lines, context_tokens
##########

########## Auxiliary function: depending on the last character of the context (i.e. where the user is),
########## determine if we are Out-Of-Word, or instead dealing with a Prefix
def check_outofword_setting(context):
    if len(context)==0:
        return True # Here we are just starting to write

    lastchar = context[-1]
    logging.info("Last character in the context: " + str(lastchar))

    outofword_setting = str(lastchar).isspace() or (regex.match('[^\P{P}-]+', str(lastchar)) is not None)
    logging.info("Out-of-word setting: " + str(outofword_setting))

    return outofword_setting
##########


########## The Language Model object
class LM_TransformerXL():

    ########## Constructor
    # Current default choices are meant for the development phase: input from text file in the same folder
    # The current version is not parameterized by language
    def __init__(self, dataset_tools_dirpath, flag_text_or_manual=True, input_filepath=".", flag_verbose=True):

        self.flag_text_or_manual = flag_text_or_manual
        self.input_filepath = input_filepath
        self.flag_verbose = flag_verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Utils.init_logging(LOGFILE, loglevel=LOGLEVEL)

        ###### We have to load the components of the pre-trained model
        if dataset_tools_dirpath == 'english':
            model_sourcedir = 'transfo-xl-wt103'
            tokenizer_sourcedir = 'transfo-xl-wt103'
        else:
            model_sourcedir = os.path.join(dataset_tools_dirpath)
            tokenizer_sourcedir = model_sourcedir # both in .../saved tools, for now

        # Load the pre-trained model tokenizer (English, with vocabulary from wikitext 103)
        self.tokenizer = tf.TransfoXLTokenizer.from_pretrained(tokenizer_sourcedir)

        # Load the pre-trained Transformer-XL model (weights)
        self.model_class = tf.TransfoXLLMHeadModel
        self.model = self.model_class.from_pretrained(model_sourcedir)
        # n: Models are now set in evaluation mode by default when instantiated with the from_pretrained() method.

        # If we have a GPU, move the model onto cuda
        # self.model.eval() # already the default when you load from_pretrained
        self.model.to(self.device)


    ########## Given a position, select the context and predict the next word
    def predict(self, input_text_manual=""):
        Utils.init_logging('temp.log')
        # Notes:
        # A) The parameter pointer_location is the position of the pointer (in characters from the start),
        #    it is relevant only if flag_pipe_or_location is set to False
        # B) If we used flag_text_or_manual=False in the constructor, then the model object accepts manual input text,
        #    it does not read it from the file, instead it is passed here.
        self.input_text_manual = input_text_manual

        # 1) Loads the input text.
        self.all_lines, self.all_text = InputFacilities.get_input(self.flag_text_or_manual,
                                                        self.input_text_manual, self.input_filepath)

        # 2) From the whole text, extract the context for the prediction
        self.context = select_context(self.all_text)
        logging.info(self.context)
        logging.info('#######')

        # 3) Given the context, retrieve the vocabuary indices and the word tokens
        self.context_lines, self.context_tokens = process_context(self.context)
        self.context_indices = self.tokenizer.encode(self.context_tokens)
        self.ctx_tensor = torch.tensor(self.context_indices, dtype=torch.long, device=self.device)
        self.context_tokens = self.tokenizer.convert_ids_to_tokens(ids=self.context_indices, skip_special_tokens=False)

        # 4) Determine if we are Out-Of-Word, or instead we are dealing with a prefix
        self.outofword_setting = check_outofword_setting(self.context)

        # 5) Compute the predictions, using the attributes & elements of this Language Model object
        self.proposed_nextwords, self.probabilities = \
            Core.get_suggestions_nextword(self.model, self.tokenizer, self.ctx_tensor,
                                          self.outofword_setting, self.context_tokens[-1])

        # 6) Logging, on CSV and graph
        if self.flag_verbose:
            RR.write_nextwords_incsv(self.context, self.proposed_nextwords, self.probabilities)
            RR.create_graphs(self.context_tokens, self.proposed_nextwords, self.probabilities)
            logging.info(self.proposed_nextwords)
            logging.info(self.probabilities)

        return self.proposed_nextwords, self.probabilities

##########
