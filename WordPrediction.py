import os
import importlib
import sys
transformer_xl = importlib.import_module('transformer-xl')
#proj_adaptive_softmax = importlib.import_module('transformer-xl.pytorch.utils.proj_adaptive_softmax')
log_uniform_sampler = importlib.import_module('transformer-xl.pytorch.utils.log_uniform_sampler')
sys.path.append('transformer-xl.pytorch.utils.proj_adaptive_softmax')
mem_transformer = importlib.import_module('transformer-xl.pytorch.mem_transformer')
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


def exe():
    model_path = os.path.join('transformer-xl', )