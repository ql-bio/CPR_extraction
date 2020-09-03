Chemical-protein Interaction Extraction via ChemicalBERT and Attention Guided Graph Convolutional Networks in Parallel
==========

The model consists of ChemicalBERT and Attention Guided Graph Convolutional Networks (AGGCN) two parallel components. We pre-train BERT on large-scale chemical interaction corpora and re-define it as ChemicalBERT to generate high-quality contextual representation, and employ AGGCN to capture syntactic graph information of the sentence. Finally, the contextual representation and syntactic graph representation are merged into a fusion layer and then fed into the fully-connected softmax layer to extract CPIs.


See below for an overview of the model architecture:

![Architecture](fig/Architecture.jpg "Architecture")

  

## Requirements

- Python 3 (tested on 3.6.8)

- PyTorch (tested on 0.4.1)

- CUDA (tested on 9.0)

- tqdm

- unzip, wget (for downloading only)


## Evaluation
we have also conducted experiments on the ChemProt corpus and the DDIExtraction 2013 corpus

Testing on CPI extraction
```
python3 eval_cpi.py
```


Testing on DDI extraction
```
python3 eval_ddi.py
```