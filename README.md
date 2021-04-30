# TCube: A Domain-Agnostic Framework for Time series Narration

This repository contains the code for the paper: "TCube: A Domain-Agnostic Framework for Time series Narration" currently under review for ACMKDD 2021.

![Alt text](Images/arch2.png?raw=true "The two stage TCube framework: In Stage I, the system extracts trends, regimes, and peaks from the input time series which is formulated into a knowledge graph. In Stage II, a PLM fine-tuned for graph-to-text generation generates the narrative from the input graph.")

The PLMs used in this effort (T5, BART, and GPT-2) are implemented using the HuggingFace library (https://huggingface.co/) and finetuned to the WebNLG v3 (https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/release_v3.0) and DART (https://arxiv.org/abs/2007.02871) datasets. 

Clones of both datasets are available under /Finetune PLMs/Datasets in this repository. 

The PLMs fine-tuned to WebNLG/DART could not be uploaded due to the 1GB limitations of GitLFS. However, pre-made scripts in this repository (detailed below) are present for convientiently fine-tuning these models.

The entire repository is based on Python 3.6 and the results are visaulized through the iPython Notebooks.

## Dependencies

### Interactive Environments
- notebook
- ipywidgets==7.5.1

### Deep Learning Frameworks
- torch 1.7.1 (suited to your CUDA version)
- pytorch-lightning 0.9.0
- transformers==3.1.0

### NLP Toolkits
- sentencepiece==0.1.91
- nltk

### Scientific Computing, Data Manipulation, and Visualizations
- numpy
- scipy
- sklearn
- matplotib
- pandas
- pwlf

### Evaluation
- rouge-score
- textstat
- lexical_diversity
- language-tool-python

### Misc
- xlrd
- tqdm
- cython

> Please make sure that the aforementioned Python packages with their specified versions are installed in your system in a separate virtual environment.

## Data-Preprocessing Scripts


