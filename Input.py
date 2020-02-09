import logging
import os
from io import open
import torch
import Utilities as Utils

### When reading from a text for inference
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

### When iterating over a corpus for training/validation
class TextCorpusIterator():
    def __init__(self, tokenizer, file_path, batch_size=4, sequence_length=16):

        assert os.path.isfile(file_path)
        self.tokenizer = tokenizer

        self.file_path = file_path
        logging.info("Dataset file at: %s", file_path)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_text_file = open(file_path, 'r')
        self.readingbuffer = []
        self.total_training_iterations = 0

    def __next__(self):
        #logging.info("Calling __next__()")
        self.input_size = self.sequence_length * self.batch_size

        while len(self.readingbuffer) < self.input_size:
            self.next_line = self.input_text_file.readline()
            if self.next_line == '':
                break # End of the dataset
            self.readingbuffer.extend(self.tokenizer.tokenize(self.next_line))

        self.batch_ids = []

        for sample_start in range(0, min((self.batch_size) * self.sequence_length, len(self.readingbuffer)), self.sequence_length):
            self.sample_tokens = self.readingbuffer[sample_start:sample_start + self.sequence_length]
            logging.debug("self.sample_tokens=" + str(self.sample_tokens))
            if len(self.sample_tokens) < self.sequence_length:
                self.padded_sample_tokens = self.sample_tokens + ([Utils.PAD_TOKEN] * (self.sequence_length - len(self.sample_tokens)) )# pad)
                self.sample = self.tokenizer.encode(self.padded_sample_tokens)
            else: # no need to pad
                self.sample = self.tokenizer.encode(self.sample_tokens)
            self.batch_ids.append(self.sample)
        self.readingbuffer = self.readingbuffer[self.input_size:]
        logging.debug('self.readingbuffer=' + str(self.readingbuffer))

        # If we are at the end of the dataset
        if len(self.batch_ids) == 0:
            raise StopIteration

        # Transposing is necessary because, in the input of Transformer-XL, the text is read along the columns not rows
        self.input = torch.tensor(self.batch_ids).t()

        return self.input



    def __len__(self):
        if self.total_training_iterations != 0:
            return self.total_training_iterations
        try:
            while True:
                _nextbatch = self.__next__()
                self.total_training_iterations = self.total_training_iterations + 1
        except StopIteration:
            # reset file reader
            self.input_text_file.close()
            self.input_text_file = open(self.file_path, 'r')
            return self.total_training_iterations

