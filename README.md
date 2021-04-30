# TCube: A Domain-Agnostic Framework for Time series Narration

This repository contains the code for the paper: "TCube: A Domain-Agnostic Framework for Time series Narration" currently under review for ACMKDD 2021.

![Alt text](Images/arch2.png?raw=true "The two stage TCube framework: In Stage I, the system extracts trends, regimes, and peaks from the input time series which is formulated into a knowledge graph. In Stage II, a PLM fine-tuned for graph-to-text generation generates the narrative from the input graph.")

![Alt text](Images/ultimate_teaser.png?raw=true "TCube Sample Narratives.")

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

Under /Finetune PLMs in this repository there are two scripts for pre-processing the WebNLG and DART datasets:
```
preprocess_webnlg.py
preprocess_dart.py
```
These scripts draw from the original datasets in /Finetune PLMs/Datasets/WebNLGv3 and /Finetune PLMs/Datasets/DART and prepare CSV files in /Finetune PLMs/Datasets breaking the original datasets into train, dev, and test sets in the format required by our PLMs.

## Fine-tuning Scripts
Under /Finetune PLMs in this repository there are three scripts for fine-tuning T5, BART, and GPT-2:
```
finetuneT5.py
finetuneBART.py
finetuneGPT2.py
```

## Visualization and Evaluation Notebooks
In the root directory are 10 notebooks.
For the descriptions of the time-series datasets used:
```
Datatsets.ipynb
```
For comparisons of segmentation and regime-change detection algorithms:
```
Error Determination.ipynb
Regime Detection.ipynb
Segmentation.ipynb
Trend Detection Plot.ipynb
```
For the evaluation of the TCube framework on respective time-series datasets:
```
T3-COVID.ipnyb
T3-DOTS.ipnyb
T3-Pollution.ipnyb
T3-Population.ipnyb
T3-Temperature.ipnyb
```
### Citation and Contact

**If any part of this code repository or the TCube framework is used in your work, please cite our paper. Thanks! **

Contact: Mandar Sharma (mandarsharma@vt.edu), First Author.
 




