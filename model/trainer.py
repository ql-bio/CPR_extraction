"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
from model.aggcn import GCNClassifier
from utils import torch_utils

import sys
sys.path.append("..")
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear


class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def load(self, filename):
        checkpoint = torch.load(filename)
        
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

def unpack_batch(batch, cuda):
    if cuda:
        inputs = [Variable(b.cuda()) for b in batch[:10]]
        labels = Variable(batch[10].cuda())
    else:
        inputs = [Variable(b) for b in batch[:10]]
        labels = Variable(batch[10])
    tokens = batch[0]
    head = batch[5]
    subj_pos = batch[6]
    obj_pos = batch[7]
    lens = batch[1].eq(0).long().sum(1).squeeze()
    return inputs, labels, tokens, head, subj_pos, obj_pos, lens


class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
        self.model.to(self.opt["device"])
        if opt['cuda']:
            self.model.cuda()

    def predict(self, batch, unsort=True):
        input_ids,input_mask,segment_ids,label_id=batch[12],batch[13],batch[14],batch[15]
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])
        l = (inputs[1].data.cpu().numpy() == 0).astype(np.int64).sum(1)
        orig_idx = batch[11]
        device=self.opt["device"]
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_id = label_id.to(device)
        self.model.eval()
        with torch.no_grad():
            logits,loss = self.model(inputs,input_ids,input_mask,segment_ids,label_id,l,b=1)
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions, probs, loss.item()
