import os
import importlib
import torch
import Input
import logging
import re
import Utilities as Utils
import string
import torch.nn.functional as F
from Utilities import DEVICE

CTX_LEN = 256 # number of tokens in a full transformer window
LOGFILE = 'Transformer_LM.log'
LOGLEVEL = logging.WARN


###### top-k filtering: top_k > 0: keep only top k tokens with highest probability
def top_k_top_p_filtering(logits, top_k=0, filter_value=-float('Inf')):
  #assert logits.dim() == 1  # batch size 1 for now
  top_k = min(top_k, logits.size(-1))  # Safety check
  if top_k > 0:
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value
  return logits


##### Modified from: transformer-xl/pytorch/utils/proj_adaptive_softmax >>
#####                ProjectedAdaptiveLogSoftmax.forward(self, hidden, target, keep_order=False)
##### Parameters: hidden :: [len*bsz x d_proj] ; target :: [len*bsz]
def get_logits(p_a_logsoftmax, hidden, target, keep_order=False):

    if hidden.size(0) != target.size(0):
        raise RuntimeError('Input and target should have the same size '
                           'in the batch dimension.')

    cutoff_values = [0] + p_a_logsoftmax.cutoffs

    if p_a_logsoftmax.n_clusters == 0:
        logit_all = p_a_logsoftmax._compute_logit(hidden, p_a_logsoftmax.out_layers[0].weight,
                                    p_a_logsoftmax.out_layers[0].bias, p_a_logsoftmax.out_projs[0])
    else:
        # construct weights and biases
        weights, biases = [], []
        for i in range(len(p_a_logsoftmax.cutoffs)):
            if p_a_logsoftmax.div_val == 1:
                l_idx, r_idx = p_a_logsoftmax.cutoff_ends[i], p_a_logsoftmax.cutoff_ends[i + 1]
                weight_i = p_a_logsoftmax.out_layers[0].weight[l_idx:r_idx]
                bias_i = p_a_logsoftmax.out_layers[0].bias[l_idx:r_idx]
            else:
                weight_i = p_a_logsoftmax.out_layers[i].weight
                bias_i = p_a_logsoftmax.out_layers[i].bias

            if i == 0:
                weight_i = torch.cat(
                    [weight_i, p_a_logsoftmax.cluster_weight], dim=0)
                bias_i = torch.cat(
                    [bias_i, p_a_logsoftmax.cluster_bias], dim=0)

            weights.append(weight_i)
            biases.append(bias_i)

        head_weight, head_bias, head_proj = weights[0], biases[0], p_a_logsoftmax.out_projs[0]

        head_logit = p_a_logsoftmax._compute_logit(hidden, head_weight, head_bias, head_proj)
        logit_all = head_logit[0:cutoff_values[1]]

        for i in range(len(cutoff_values) - 1):
            l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

            mask_i = (target >= l_idx) & (target < r_idx) # the target must be in one of the sections after the first
            indices_i = mask_i.nonzero().squeeze()
            print("indices_i = " + str(indices_i))

            if indices_i.numel() == 0:
                continue

            if i == 0:
                pass # logprob_i = head_logprob_i.gather(1, target_i[:,None]).squeeze(1)
            else:
                weight_i, bias_i, proj_i = weights[i], biases[i], p_a_logsoftmax.out_projs[i]

                hidden_i = hidden.index_select(0, indices_i)

                tail_logit_i = p_a_logsoftmax._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                # must still refine
                print("tail_logit_i.shape = " + str(tail_logit_i.shape))
                logit_all = torch.cat([logit_all, tail_logit_i], dim=0)

    return logit_all


##### Modified from: transformer-xl/pytorch/mem_transformer >> MemTransformerLM.forward(self, data, target, *mems)
def get_probabilities(txl_model, data, target, *mems):
    # nn.DataParallel does not allow size(0) tensors to be broadcasted.
    # So, have to initialize size(0) mems inside the model forward.
    # Moreover, have to return new_mems to allow nn.DataParallel to piece
    # them together.
    if not mems: mems = txl_model.init_mems()

    tgt_len = target.size(0)
    txl_model.eval()# evaluation mode
    with torch.no_grad():
        hidden, new_mems = txl_model._forward(data, mems=mems)

    pred_hid = hidden[-tgt_len:]

    projected_adaptive_logsoftmax = txl_model.crit

    logits = get_logits(projected_adaptive_logsoftmax, pred_hid, target)

    logits = logits.squeeze()

    filtered_logits = top_k_top_p_filtering(logits, top_k=10)

    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    top_probs = sorted_probs[0:10]
    top_indices = sorted_indices[0:10]

    loss = txl_model.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
    # loss = loss.view(tgt_len, -1)

    return top_probs, top_indices


########## Entry point function
def predict(model, vocabulary, context_tokens, labels_shape=(1,1)):
    inverse_example = "<unk> is an English film, television and theatre actor . He had"
    ctx_tensor = vocabulary.convert_to_tensor(context_tokens).t().to(torch.int64).to(DEVICE)

    labels= torch.ones(size=(1,1)).to(torch.int64).to(DEVICE) # placeholder label, to specify that we are predicting the next token (or possibly more)
    top_probs, top_indices = get_probabilities(model, data=ctx_tensor, target=labels)

    print(top_probs)
    print(top_indices)

    # we have a choice. To visualize it, go from vocabulary indices to words:
    top_words = list(
        map(lambda i_tensor: vocabulary.convert_to_sent([i_tensor.item()]),  # , skip_special_tokens=True
            top_indices))

    return top_words, top_probs