import numpy as np
import torch
import string
import transformers as tf
import torch.nn.functional as torchF
import logging
#import TW_Utilities as TW_Utils

NUM_SUGGESTIONS=10
TOPK_DIRECT = 100 #the top-k tokens with the highest probability
TOPK_FORPREFIXSEARCH = 10000

########## Get a list of 10 suggestions for the next word. Eliminates punctuation
def get_suggestions_nextword(model, tokenizer, context_tensor, oow_flag, prefix):
    context = context_tensor.unsqueeze(0).repeat(1, 1)
    with torch.no_grad():
        inputs = {'input_ids': context}
        outputs = model(**inputs)

        next_token_logits = outputs[0][0, -1, :]
        if oow_flag:
            top_k = TOPK_DIRECT
        else:
            top_k = TOPK_FORPREFIXSEARCH

        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k)

        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
        sorted_probs = torchF.softmax(sorted_logits, dim=-1)
        top_probs = sorted_probs[0:top_k]

        # we have a choice. To visualize it, go from vocabulary indices to words:
        top_words = list(
            map(lambda i_tensor: tokenizer.convert_ids_to_tokens(i_tensor.item()), # , skip_special_tokens=True
                sorted_indices[0:top_k]))

        # If we are in an in-word setting, we must keep only the candidates that begin with the given prefix
        if (not oow_flag):
            top_words, top_probs = keep_only_prefix_words(top_words, top_probs, prefix)
        # Filter out punctuation and <unk>:
        top_words, top_probs = eliminate_punctuation_specials(top_words, top_probs)

        logging.info("Number of most likely candidates, eiminating <eos> and punctuation: " + str(len(top_words)))

        selected_words = top_words[0:NUM_SUGGESTIONS]
        selected_probs = top_probs[0:NUM_SUGGESTIONS]

    return selected_words, selected_probs

############
def top_k_top_p_filtering(logits, top_k=0, filter_value=-float('Inf')):
  """    Args:
          logits: logits distribution shape (vocabulary size)
          top_k > 0: keep only top k tokens with highest probability (top-k filtering).  """
  assert logits.dim() == 1  # batch size 1 for now
  top_k = min(top_k, logits.size(-1))  # Safety check
  if top_k > 0:
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value
  return logits


############
def eliminate_punctuation_specials(words, probs):
    logging.info("words=" + str(words))
    indices_to_keep = []
    for i in range(0, len(words)):
        word_set = {str(words[i])}
        if word_set.issubset(set(string.punctuation)) or \
           word_set.issubset(set(string.whitespace)) or \
           word_set.issubset({'<eos>', '<unk>'}):
            logging.info('Removed: ' + str(words[i]))
            continue
        else:
            indices_to_keep.append(i)

    words_to_keep = np.array(words)[np.array(indices_to_keep)]
    probs_to_keep = np.array(probs)[np.array(indices_to_keep)]

    return words_to_keep, probs_to_keep

##########
def keep_only_prefix_words(words, probs, prefix):

  indices_to_keep = []
  for i in range(0, len(words)):
    if words[i].startswith(prefix):
      indices_to_keep.append(i)
    else:
      continue

  words_to_keep = np.array(words)[np.array(indices_to_keep)]
  probs_to_keep = np.array(probs)[np.array(indices_to_keep)]

  return words_to_keep, probs_to_keep