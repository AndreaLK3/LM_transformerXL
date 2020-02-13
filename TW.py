import os
import importlib
import torch
import Input
import logging
import re
import Utilities as Utils
import regex
import sys
import RecordResults as Rr
import WordPredictionCore as Wpc
from Utilities import DEVICE


CTX_LEN = 256  # number of tokens in a full transformer window
LOGFILE = 'LM.log'
LOGLEVEL = logging.WARN

########## The Language Model object
class LM_TransformerXL():

    ########## Constructor
    # Current default choices are meant for the development phase: input from text file in the same folder
    # The current version is not parameterized by language
    def __init__(self, dataset, flag_text_or_manual=True, input_filepath=".", flag_verbose=True):

        self.dataset_dirpath = (dataset.value)[0]
        self.flag_text_or_manual = flag_text_or_manual
        self.input_filepath = input_filepath
        self.flag_verbose = flag_verbose

        Utils.init_logging(LOGFILE, loglevel=LOGLEVEL)

        ###### We have to load the components of the pre-trained model
        self.model, self.vocabulary = load_model_and_vocab(self.dataset_dirpath)

        # If we have a GPU, move the model onto cuda
        # self.model.eval() # already the default when you load from_pretrained
        self.model.to(DEVICE)


    ########## Given a position, select the context and predict the next word
    def predict(self, input_text_manual=""):
        Utils.init_logging('temp.log')
        # If we used flag_text_or_manual=False in the constructor, the model object accepts manual input text

        # 1) Loads the input text.
        self.all_lines, self.all_text = Input.get_input(self.flag_text_or_manual,
                                                        input_text_manual, self.input_filepath)

        # 2) From the whole text, extract the context for the prediction
        self.context = select_context(self.all_text)

        # 3) Tokenize the context
        self.context_tokens = self.vocabulary.tokenize(self.context)
        logging.info('*********\n' + str(self.context_tokens) + '\n*********')

        # 4) Determine if we are Out-Of-Word, or instead we are dealing with a prefix
        self.outofword_setting = check_outofword_setting(self.context)

        # 5) Compute the predictions, using the attributes & elements of this Language Model object
        self.proposed_nextwords, self.probabilities = Wpc.predict(self.model, self.vocabulary, self.context_tokens)

        # 6) Logging, on CSV and graph
        if self.flag_verbose:
            Rr.write_nextwords_incsv(self.context, self.proposed_nextwords, self.probabilities)
            Rr.create_graphs(self.context_tokens, self.proposed_nextwords, self.probabilities.cpu().tolist())
            logging.info(self.proposed_nextwords)
            logging.info(self.probabilities)


########## Functions to load the model and the vocabulary ##########

def load_model_and_vocab(dataset_dirpath):

    sys.path.append(os.path.join('transformer-xl', 'pytorch'))
    data_utils = importlib.import_module('data_utils')

    corpus_fpath = os.path.join(dataset_dirpath, 'corpus.pt')
    if os.path.exists(corpus_fpath):
        text_corpus = torch.load(corpus_fpath)
    else:
        dataset_type = 'wt103'  # the default processing (e.g. <eos>) that we are using, among the alternatives
        text_corpus = data_utils.get_lm_corpus(dataset_dirpath, dataset_type)
        torch.save(text_corpus, corpus_fpath)

    vocabulary = text_corpus.vocab
    vocabulary.unk_idx = vocabulary.sym2idx['<unk>']

    txl_model = get_txl_model(dataset_dirpath)

    return txl_model, vocabulary


### If we have already a trained t-xl model, load it. Else, declare that we must train it.
def get_txl_model(dataset_dirpath):

    Utils.create_folders_ifneeded([dataset_dirpath])
    model_fpath = os.path.join(dataset_dirpath, 'model.pt')

    if os.path.exists(model_fpath):
        # adjusting the sys.path to allow us to load a model that is not in transformer-xl/pytorch
        sys.path.append(os.path.join(os.getcwd(), 'transformer-xl', 'pytorch'))
        sys.path.append(os.path.join(os.getcwd(), 'transformer-xl', 'pytorch', 'utils'))
        txl_model = torch.load(model_fpath)
        return txl_model
    else:
        logging.info("Model not present. We must train it on the specified dataset and select the best version")
        return None




########## Functions to select the context in the input text, and define the Out-of-word setting ##########

### Auxiliary function: from the whole text, extract the context for the prediction
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
###


### Auxiliary function: depending on the last character of the context (i.e. where the user is),
### determine if we are Out-Of-Word, or instead dealing with a Prefix
def check_outofword_setting(context):
    if len(context)==0:
        return True # Here we are just starting to write

    lastchar = context[-1]
    logging.info("Last character in the context: " + str(lastchar))

    outofword_setting = str(lastchar).isspace() or (regex.match('[^\P{P}-]+', str(lastchar)) is not None)
    logging.info("Out-of-word setting: " + str(outofword_setting))

    return outofword_setting
##########

