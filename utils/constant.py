# # CPI
EMB_INIT_RANGE = 1.0

 # vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'CHEMICAL': 2}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'GENE-N': 2, 'GENE-Y': 3}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'CHEMICAL': 3, 'GENE-N': 4, 'GENE-Y': 5}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'DT': 2, 'NN': 3, 'IN': 4, 'CC': 5, 'CD': 6, 'NNS': 7, ':': 8, 'VBG': 9, ',': 10, 'RB': 11, 'VBD': 12, 'VBN': 13, 'PRP$': 14, 'NNP': 15, 'JJ': 16, '.': 17, 'FW': 18, 'VBZ': 19, 'TO': 20, 'VB': 21, 'PRP': 22, 'VBP': 23, 'MD': 24, '``': 25, "''": 26, 'SYM': 27, 'EX': 28, 'WDT': 29, 'RP': 30, 'RBS': 31, 'PDT': 32, 'JJR': 33, 'LS': 34, 'POS': 35, 'JJS': 36, 'WRB': 37, 'RBR': 38, 'NNPS': 39, 'WP': 40, 'WP$': 41}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'det': 2, 'nsubj': 3, 'prep': 4, 'amod': 5, 'pobj': 6, 'cc': 7, 'dep': 8, 'conj': 9, 'punct': 10, 'root': 11, 'dobj': 12, 'advmod': 13, 'mwe': 14, 'nsubjpass': 15, 'vmod': 16, 'auxpass': 17, 'poss': 18, 'nn': 19, 'num': 20, 'appos': 21, 'rcmod': 22, 'aux': 23, 'cop': 24, 'xcomp': 25, 'parataxis': 26, 'mark': 27, 'ccomp': 28, 'pcomp': 29, 'number': 30, 'expl': 31, 'acomp': 32, 'neg': 33, 'npadvmod': 34, 'prt': 35, 'iobj': 36, 'preconj': 37, 'advcl': 38, 'quantmod': 39, 'possessive': 40, 'csubj': 41, 'tmod': 42, 'predet': 43, 'csubjpass': 44, 'discourse': 45}
NEGATIVE_LABEL = 'false'

LABEL_TO_ID = {'false': 0, 'CPR:4': 1, 'CPR:3': 2, 'CPR:9': 3, 'CPR:6': 4, 'CPR:5': 5}

INFINITY_NUMBER = 1e12


# # DDI
# EMB_INIT_RANGE = 1.0

# # vocab
# PAD_TOKEN = '<PAD>'
# PAD_ID = 0
# UNK_TOKEN = '<UNK>'
# UNK_ID = 1

# VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# # hard-coded mappings from fields to ids
# SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'drug': 2, 'group': 3,'brand':4,'drug_n':5}

# OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'drug': 2, 'group': 3,'brand':4,'drug_n':5}

# NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'drug': 3, 'group': 4, 'brand': 5, 'drug_n': 6}

# POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NN': 2, 'NNS': 3, 'VBD': 4, 'TO': 5, 'VB': 6, 'JJ': 7, 'IN': 8, 'DT': 9, 'VBG': 10, ',': 11, 'VBP': 12, 'CC': 13, '.': 14, 'WRB': 15, 'VBN': 16, 'RB': 17, 'FW': 18, 'MD': 19, 'JJR': 20, 'VBZ': 21, ':': 22, 'NNP': 23, 'WDT': 24, 'CD': 25, 'SYM': 26, 'PRP': 27, 'PRP$': 28, 'RBS': 29, 'EX': 30, 'WP': 31, 'WP$': 32, 'NNPS': 33, 'RBR': 34, '``': 35, "''": 36, 'PDT': 37, 'LS': 38, 'POS': 39, 'JJS': 40, 'RP': 41}

# DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'amod': 2, 'nn': 3, 'nsubj': 4, 'root': 5, 'aux': 6, 'cop': 7, 'xcomp': 8, 'prep': 9, 'pobj': 10, 'mark': 11, 'det': 12, 'advcl': 13, 'punct': 14, 'conj': 15, 'cc': 16, 'advmod': 17, 'nsubjpass': 18, 'auxpass': 19, 'rcmod': 20, 'pcomp': 21, 'dobj': 22, 'dep': 23, 'mwe': 24, 'neg': 25, 'appos': 26, 'vmod': 27, 'ccomp': 28, 'parataxis': 29, 'num': 30, 'poss': 31, 'iobj': 32, 'npadvmod': 33, 'discourse': 34, 'quantmod': 35, 'expl': 36, 'acomp': 37, 'number': 38, 'preconj': 39, 'csubj': 40, 'tmod': 41, 'csubjpass': 42, 'predet': 43, 'possessive': 44, 'prt': 45}

# NEGATIVE_LABEL = 'false'

# LABEL_TO_ID={'false': 0, 'effect': 1, 'mechanism': 2, 'advise': 3, 'int': 4}

# INFINITY_NUMBER = 1e12
