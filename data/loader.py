"""
Data loader for TACRED json files.
"""
import sys
sys.path.append("..")
import json
import random
import torch
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils import constant
class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        tokenizer=BertTokenizer.from_pretrained(opt["bert_model_file"], do_lower_case=True)
        data = self.preprocess(data, vocab, opt,tokenizer,evaluation)

        #shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-1]] for d in data]
        
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt,tokenizer,evaluation):
        """ Preprocess the data and convert to ids. """
        processed = []
        if not evaluation:
            data = sorted(data, key=lambda f: len(f["token"]))
        for d in data:
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['entity1_type']] * (se-ss+1)
            tokens[os:oe+1] = ['OBJ-'+d['entity2_type']] * (oe-os+1)
            tokens = map_to_ids(tokens, vocab.word2id)
            
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            #判断head中是否有0，若没有0则出错
            assert any([x == 0 for x in head])
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            subj_type = [constant.SUBJ_NER_TO_ID[d['entity1_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['entity2_type']]]
            relation = self.label2id[d['relation']]
            input_ids,input_mask,segment_ids,label_id=getBertbacth(list(d['token']),opt,d['subj_start'],d['subj_end'],d['obj_start'],d['obj_end'],tokenizer,relation)
            processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type,obj_type,relation,input_ids,input_mask,segment_ids,label_id)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))

        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = get_long_tensor(batch[4], batch_size)
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)
        rels = torch.LongTensor(batch[9])
        input_ids=torch.tensor(batch[10], dtype=torch.long)
        input_mask=torch.tensor(batch[11], dtype=torch.long)
        segment_ids=torch.tensor(batch[12], dtype=torch.long)
        label_id=torch.tensor(batch[13], dtype=torch.long)
        return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, rels, orig_idx,input_ids,input_mask,segment_ids,label_id)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def getBertbacth(tokensss,opt,subj_start,subj_end,obj_start,obj_end,tokenizer,relation):
    tokenss = [convert_token(token11) for token11 in tokensss]
    max_seq_length=opt["max_seq_length"]
    special_tokens={}
    def get_special_token(w):
        if w not in special_tokens:
            special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
        return special_tokens[w]
    SUBJECT_START = get_special_token("SUBJ_START")
    SUBJECT_END = get_special_token("SUBJ_END")
    OBJECT_START = get_special_token("OBJ_START")
    OBJECT_END = get_special_token("OBJ_END")
    tokens = ["[CLS]"]
    for i, token in enumerate(tokenss):
        if i == subj_start:
            tokens.append(SUBJECT_START)
        if i == obj_start:
            tokens.append(OBJECT_START)
        if (i >= subj_start) and (i <= subj_end):
            pass
        elif (i >=obj_start ) and (i <=obj_end ):
            pass
        else:
            for sub_token in tokenizer.tokenize(token):
                tokens.append(sub_token)
    tokens.append("[SEP]")
    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    label_id=relation
    return input_ids,input_mask,segment_ids,label_id
    
def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
            return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token
    
def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]