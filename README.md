# gae
Graph Auto-Encoder in PyTorch

This is a PyTorch implementation of the (Variational) Graph Auto-Encoder model described in the paper:
 
T. N. Kipf, M. Welling, [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308), NIPS Workshop on Bayesian Deep Learning (2016)

The code in this repo is based on https://github.com/zfjsail/gae-pytorch which only implements the VGAE
and refers to the original TensorFlow GAE implementation from https://github.com/tkipf/gae.

### Requirements
The code has been written with the following specifications, older versions might apply:
- Python 3.11.9
- NetworkX 3.2.1
- NumPy 1.26.3
- PyTorch 2.2.0
- scikit-learn 1.12.0
- SciPy 1.12.0

Install requirements via ```pip install -r requirements.txt``` 

### How to run
```bash
python gae/train.py
```
