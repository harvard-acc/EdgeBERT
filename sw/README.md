# EdgeBERT

This repository contains the software code (modifed from [HuggingFace](https://github.com/huggingface/transformers)) used to train and evaluate models in the paper:

**EdgeBERT: Sentence-Level Energy Optimizations for Latency-Aware Multi-Task NLP Inference** (https://arxiv.org/abs/2011.14203).

# Installation Instructions:
System requirements (assuming a unix system)
* Anaconda3 (tested with version 5.0.1)
* Cuda (tested with version 10.0.130)
* Cudnn (tested with version 7.4.1.5)

Run the following commands:

```
conda create --name test-edge python=3.7
source activate test-edge

pip install torch
pip install tensorboard
pip install -U scikit-learn

cd EdgeBERT/transformers
python setup.py install
```
# Producing a lookup table for entropy prediction
To produce a lookup table for entropy prediction:

1. Open Entropy_LUT/entropypredictor.ipynb in Google Colab and load the desired training and test entropy datasets.

2. Run all cells in this notebook and download the resulting csv file.

# Training and evaluating models
Change into the EdgeBERT/scripts directory and follow the steps in the README.md file.

# Alternative: Jupyter notebook for training and evaluating models
The edgebert.ipynb notebook allows for training and evaluating models in Google Colab. Note that training times are such that training cannot be run using a hosted runtime.
