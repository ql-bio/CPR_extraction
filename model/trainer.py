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

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer2, new_lr)

    def load(self, filename):
        checkpoint = torch.load(filename)
        
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']


    def save(self, filename, opt):
        params = {
                'model': self.model.state_dict(),
                'config': opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


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
        self.criterion = nn.CrossEntropyLoss()
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters1=[
                {'params': [p for n, p in param_optimizer
                            if (not any(nd in n for nd in no_decay)) and "bert" in n[0:11]], 'weight_decay': 0.01,'lr': 1e-6},
                {'params': [p for n, p in param_optimizer
                            if (any(nd in n for nd in no_decay)) and "bert" in n[0:11]], 'weight_decay': 0.0,'lr': 1e-6}
            ]
        optimizer_grouped_parameters2=[
                {'params': [p for n, p in param_optimizer if "gcn_model" in n[0:16]]},
                {'params': [p for n, p in param_optimizer if "classifier" in n[0:17]]}
            ]
        num_train_optimization_steps = opt["train_batch_len"] // 1 * opt["num_epoch"]
        self.optimizer1 = BertAdam(optimizer_grouped_parameters1,warmup=0.1,t_total=num_train_optimization_steps,lr=1e-6)
        self.optimizer2 = torch.optim.SGD(optimizer_grouped_parameters2, lr=opt["lr"], weight_decay=0)

        

    def update(self, batch):
        input_ids,input_mask,segment_ids,label_id=batch[12],batch[13],batch[14],batch[15]
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])
        l = (inputs[1].data.cpu().numpy() == 0).astype(np.int64).sum(1)
        # step forward
        self.model.train()
        device=self.opt["device"]
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_id = label_id.to(device)
        # logits, pooling_output = self.model(inputs,input_ids,input_mask,segment_ids,label_id,l)
        loss = self.model(inputs,input_ids,input_mask,segment_ids,label_id,l,b=None)
        loss_val = loss.item()
        # backward
        loss.backward()
        
        parameters=[p for n,p in self.model.named_parameters() if "bert.bert" not in n[0:9]]
        torch.nn.utils.clip_grad_norm_(parameters , self.opt['max_grad_norm'])
        self.optimizer1.step()
        self.optimizer1.zero_grad()
        self.optimizer2.step()
        self.optimizer2.zero_grad()
        return loss_val

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
        # label_id = None
        # forward
        self.model.eval()
        with torch.no_grad():
            logits,loss = self.model(inputs,input_ids,input_mask,segment_ids,label_id,l,b=1)
        # loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        # self.model.train()
        return predictions, probs, loss.item()
