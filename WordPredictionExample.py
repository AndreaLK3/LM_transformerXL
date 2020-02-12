import os
import importlib

#transformer_xl = importlib.import_module('transformer-xl')
#proj_adaptive_softmax = importlib.import_module('transformer-xl.pytorch.utils.proj_adaptive_softmax')
#log_uniform_sampler = importlib.import_module('transformer-xl.pytorch.utils.log_uniform_sampler')
# mem_transformer = importlib.import_module('transformer-xl.pytorch.mem_transformer')
import torch.nn.functional as F
import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

############
def top_k_top_p_filtering(logits, top_k=0, filter_value=-float('Inf')):
  """    Args:
          logits: logits distribution shape (vocabulary size)
          top_k > 0: keep only top k tokens with highest probability (top-k filtering).  """
  #assert logits.dim() == 1  # batch size 1 for now
  top_k = min(top_k, logits.size(-1))  # Safety check
  if top_k > 0:
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value
  return logits


def get_logits(p_a_logsoftmax, hidden, target, keep_order=False):

    '''
        hidden :: [len*bsz x d_proj]
        target :: [len*bsz]
    '''

    if hidden.size(0) != target.size(0):
        raise RuntimeError('Input and target should have the same size '
                           'in the batch dimension.')

    print("p_a_logsoftmax.n_clusters =" + str(p_a_logsoftmax.n_clusters))

    cutoff_values = [0] + p_a_logsoftmax.cutoffs
    print("cutoff_values = " + str(cutoff_values))

    if p_a_logsoftmax.n_clusters == 0:
        logit_all = p_a_logsoftmax._compute_logit(hidden, p_a_logsoftmax.out_layers[0].weight,
                                    p_a_logsoftmax.out_layers[0].bias, p_a_logsoftmax.out_projs[0])
        # nll = -F.log_softmax(logit, dim=-1) \
        #         .gather(1, target.unsqueeze(1)).squeeze(1)
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
        print("head_logit.shape = " + str(head_logit.shape))
        logit_all = head_logit[0:cutoff_values[1]]

        head_logprob = F.log_softmax(head_logit, dim=1)

        nll = torch.zeros_like(target,
                dtype=hidden.dtype, device=hidden.device)

        offset = 0

        for i in range(len(cutoff_values) - 1):
            l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

            mask_i = (target >= l_idx) & (target < r_idx) # the target must be in one of the sections after the first
            indices_i = mask_i.nonzero().squeeze()
            print("indices_i = " + str(indices_i))

            if indices_i.numel() == 0:
                continue

            target_i = target.index_select(0, indices_i) - l_idx
            head_logprob_i = head_logprob.index_select(0, indices_i)

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



def get_probabilities(txl_model, data, target, *mems):
    # nn.DataParallel does not allow size(0) tensors to be broadcasted.
    # So, have to initialize size(0) mems inside the model forward.
    # Moreover, have to return new_mems to allow nn.DataParallel to piece
    # them together.
    if not mems: mems = txl_model.init_mems()

    print("data.shape=" + str(data.shape))
    tgt_len = target.size(0)
    print("tgt_len=" + str(tgt_len))
    with torch.no_grad():
        hidden, new_mems = txl_model._forward(data, mems=mems)

    pred_hid = hidden[-tgt_len:]

    #print(pred_hid)
    print('pred_hid.shape=' + str(pred_hid.shape))
    # # if txl_model.sample_softmax > 0 and txl_model.training:
    # # assert txl_model.tie_weight #the default option is to have tied weights, even when not specified
    # logit = log_uniform_sampler.sample_logits(txl_model.word_emb,
    #                       txl_model.out_layer.bias, target, pred_hid, txl_model.sampler)
    # probabilities = F.softmax(logit)
    # print(probabilities.shape)
    # #loss = -F.log_softmax(logit, -1)[:, :, 0]
    # else:
    #     loss = txl_model.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
    #     loss = loss.view(tgt_len, -1)
    #loss = txl_model.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
    #print(loss)
    #print(loss.shape)

    # In the MemTransformer,
    # p_a_logsoftmax.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
    #                                         cutoffs, div_val=div_val)
    projected_adaptive_logsoftmax = txl_model.crit

    print("hidden.shape="+str(hidden.shape))
    print("target.shape" + str(target.shape))
    logits = get_logits(projected_adaptive_logsoftmax, pred_hid, target)

    print("logits.shape="+ str(logits.shape))
    logits = logits.squeeze()
    print("logits.shape="+ str(logits.shape))

    #projected_adaptive_logsoftmax.cutoffs
    filtered_logits = top_k_top_p_filtering(logits, top_k=10)

    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    top_probs = sorted_probs[0:10]
    top_indices = sorted_indices[0:10]

    print("Checking: view & loss...")
    #print("pred_hid.view(-1, pred_hid.size(-1))=\n" + str(pred_hid.view(-1, pred_hid.size(-1))))
    print("pred_hid.view(-1, pred_hid.size(-1)).shape=" + str(pred_hid.view(-1, pred_hid.size(-1)).shape))
    print("target.view(-1)="+str(target))

    loss = txl_model.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
    loss = loss.view(tgt_len, -1)
    print("loss="+str(loss))

    return top_probs, top_indices




def exe_wt2():
    os.chdir(os.path.join('transformer-xl', 'pytorch'))
    data_utils = importlib.import_module('data_utils')
    txl_model = torch.load('model.pt')

    os.chdir(os.path.join('..','..'))

    datadir = os.path.join('transformer-xl','data', 'wikitext-2')
    dataset = 'wt103' # the default processing that we are using, among a limited number of choices
    text_corpus = data_utils.get_lm_corpus(datadir, dataset)
    vocabulary = text_corpus.vocab

    vocabulary.unk_idx =vocabulary.sym2idx['<unk>']

    example = [4, 24, 31, 362, 110, 2, 486, 6, 3008, 1652, 3, 59, 32]
    s1 = vocabulary.convert_to_sent(example)
    print(example)
    print(s1)

    print("inverse=")
    inverse_example = "<unk> is an English film, television and theatre actor . He had"
    t2 = vocabulary.convert_to_tensor(vocabulary.tokenize(inverse_example))
    print(t2)

    example_input = torch.tensor([[4, 24, 31, 362, 110, 2, 486, 6, 3008, 1652, 3, 59]]).t().to(torch.int64).to(DEVICE)
    example_labels = torch.tensor([[32]]).to(torch.int64).to(DEVICE)


    top_probs, top_indices = get_probabilities(txl_model, data=example_input, target=example_labels)

    print(top_probs)
    print(top_indices)

    # we have a choice. To visualize it, go from vocabulary indices to words:
    top_words = list(
        map(lambda i_tensor: vocabulary.convert_to_sent([i_tensor.item()]),  # , skip_special_tokens=True
            top_indices))

    print(top_words)


