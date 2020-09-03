"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch
import os
from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default="best_model/CPI_model",help='Directory of the model.')
parser.add_argument('--data_dir', type=str, default='dataset/CPI')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--dataset', type=str, default='test.json', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')

parser.add_argument('--bert_model_file', type=str,default="best_model/ChemicalBERT", help='Filename of the pretrained model.')
args = parser.parse_args()
import numpy as np
torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
model_file=os.path.join(args.model_dir,args.model)
opt = torch_utils.load_config(model_file)

# load vocab
vocab_file = args.data_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)

# load data
data_file= os.path.join(args.data_dir,args.dataset)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

opt["bert_model_file"]=args.bert_model_file
batch = DataLoader(data_file, 512, opt, vocab, evaluation=True)

print("Loading model from {}".format(model_file))
trainer = GCNTrainer(opt)
trainer.load(model_file)

predictions = []
all_probs = []
batch_iter = tqdm(batch)

for i, b in enumerate(batch_iter):
    preds, probs, _ = trainer.predict(b)
    predictions += preds
    all_probs += probs
predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)