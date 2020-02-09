import sys, os
from io import open
import numpy as np
import torch
import argparse
import transformers as pt
import Filesystem as F
import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--max_ctx_len', type=int, default=256, help='')
parser.add_argument('--max_gen_len', type=int, default=32, help='')
parser.add_argument('--topk', type=int, default=40, help='')
parser.add_argument('--start_idx', type=int, default=0, help='') #-1
parser.add_argument('--out_path', type=str, default=os.path.join('seqgen','SequenceGen_output.txt'), help='')
parser.add_argument('--inp_path', type=str, default='seqgen')
args = parser.parse_args()

def format_text(tokens):
  line = ''
  for token in tokens:
    if token == '<eos>':
      line += '\n'
    else:
      line += token
      line += ' '

  # simple rules of detokenization
  line = line.replace(' @-@ ', '-')
  line = line.replace(' @,@ ', ',')
  line = line.replace(' @.@ ', '.')
  line = line.replace(' . ', '. ')
  line = line.replace(' , ', ', ')
  line = line.replace(' : ', ': ')
  line = line.replace(' ; ', '; ')
  line = line.replace(" 's ", "'s ")
  line = line.replace(' ( ', ' (')
  line = line.replace(' ) ', ') ')

  return line

# Load pre-trained model tokenizer (vocabulary from wikitext 103)
tokenizer = pt.TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
for idx, sym in enumerate(tokenizer.idx2sym):
  tokenizer.idx2sym[idx] = sym.encode('latin1').decode('utf-8')

with open(os.path.join(args.inp_path,'test.txt'), 'r', encoding='utf-8') as f:
  lines = [l.strip().split() + ['<eos>'] for l in f.readlines()]

# Randomly choose some lines
num_lines = len(lines)

context, reference = [], []

if args.start_idx < 0:
  args.start_idx = np.random.randint(0, num_lines - 40)

idx = args.start_idx
while idx < num_lines:
  context += lines[idx]
  idx += 1
  if len(context) >= args.max_ctx_len:
    break

while idx < num_lines:
  reference += lines[idx]
  idx += 1
  if len(reference) >= args.max_gen_len:
    break

while len(context) > args.max_ctx_len:
  reference.insert(0, context.pop())

# Convert token to vocabulary indices
ctx_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(context)])

# Load pre-trained model (weights)
mymodel_dir = os.path.join(F.DIR_PROCESSED_TEXT, 'Wiki_Dansk', F.DIR_SAVED_TOOLS)
model = pt.TransfoXLLMHeadModel.from_pretrained(mymodel_dir)
model.eval()

# If you have a GPU, put everything on cuda
if torch.cuda.is_available():
    ctx_tensor = ctx_tensor.to('cuda')
    model.to('cuda')

unk_id = tokenizer.convert_tokens_to_ids(['<unk>'])[0]

with torch.no_grad():
  # Predict all tokens
  tensor = ctx_tensor
  generation = []
  for i in range(args.max_gen_len):
    if i == 0:
      log_prob, mems = model(tensor)
    else:
      log_prob, mems = model(tensor, mems=mems)

    prob = torch.exp(log_prob[0, -1, :])
    prob[unk_id].data.fill_(0.)

    # sample from the top-k tokens
    top_prob, top_index = torch.topk(prob, args.topk)
    token = torch.multinomial(top_prob, 1)
    token = top_index[token]

    tensor = token.detach().view(1, 1)

    symbol = tokenizer._convert_id_to_token(token.item())

    generation.append(symbol)

with open(args.out_path, 'w', encoding='utf-8') as f:
  f.write('Start line: {}'.format(args.start_idx) + '\n')
  f.write('Context len: {}'.format(len(context)) + '\n')
  f.write('-' * 80 + '\n')
  f.write('INPUT CONTEXT: ' + '\n')
  f.write(format_text(context) + '\n')
  f.write('-' * 80 + '\n')
  f.write('GENERATED SEQUENCE: ' + '\n')
  f.write(format_text(generation) + '\n')
  f.write('-' * 80 + '\n')
  f.write('REFERENCE SOLUTION: ' + '\n')
  f.write(format_text(reference[:args.max_gen_len]) + '\n')