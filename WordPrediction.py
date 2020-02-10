import os
import importlib
import Input
transformer_xl = importlib.import_module('transformer-xl')
#proj_adaptive_softmax = importlib.import_module('transformer-xl.pytorch.utils.proj_adaptive_softmax')
log_uniform_sampler = importlib.import_module('transformer-xl.pytorch.utils.log_uniform_sampler')

# mem_transformer = importlib.import_module('transformer-xl.pytorch.mem_transformer')
import torch.nn.functional as F
import torch

def forward(txl_model, data, target, *mems):
    # nn.DataParallel does not allow size(0) tensors to be broadcasted.
    # So, have to initialize size(0) mems inside the model forward.
    # Moreover, have to return new_mems to allow nn.DataParallel to piece
    # them together.
    if not mems: mems = txl_model.init_mems()

    tgt_len = target.size(0)
    hidden, new_mems = txl_model._forward(data, mems=mems)

    pred_hid = hidden[-tgt_len:]
    # if txl_model.sample_softmax > 0 and txl_model.training:
    assert txl_model.tie_weight
    logit = log_uniform_sampler.sample_logits(txl_model.word_emb,
                          txl_model.out_layer.bias, target, pred_hid, txl_model.sampler)
    probabilities = F.softmax(logit)
    print(probabilities.shape)
    #loss = -F.log_softmax(logit, -1)[:, :, 0]
    # else:
    #     loss = txl_model.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
    #     loss = loss.view(tgt_len, -1)

    return probabilities


def exe_wt2():
    os.chdir(os.path.join('transformer-xl', 'pytorch'))
    data_utils = importlib.import_module('data_utils')
    txl_model = torch.load('model.pt')
    os.chdir(os.path.join('..','..'))

    datadir = os.path.join('transformer-xl','data', 'wikitext-2')
    dataset = 'wt2'
    text_corpus = data_utils.get_lm_corpus(datadir, dataset)
    vocabulary = text_corpus.vocab

    vocabulary.unk_idx =vocabulary.sym2idx['<unk>']

    example = [4, 24, 31, 362, 110]
    s1 = vocabulary.convert_to_sent(example)
    print(example)
    print(s1)

    print("inverse=")
    inverse_example = "<unk> is an English film"
    t2 = vocabulary.convert_to_tensor(vocabulary.tokenize(inverse_example))
    print(t2)


