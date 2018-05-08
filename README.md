# MyResearch
We propose new method for feature learning from weakly-labeled data based on [StyleNet](http://hi.cs.waseda.ac.jp/~esimo/publications/SimoSerraCVPR2016.pdf) architecture.

# Codes
- `train.py`: train the model on CPU.
- `load_model.py`: model defined on it.
- `image_sampling.py`: triplet sampling.
- `model.pth`: ours model parameters.
- `convert_torch.py`: convert Torch7 model to PyTorch.

# Experimental Result
## Classification Task
We evaluate our model by classification task on Hipster Wars dataset which includes 5 class and 1893 images.

### Accuracy Result
- StyleNet Joint:       74.93 ± 0.45% (We're unable to reproduce the result of 75.9% in the paper)
- Ours Ranking(>0.83):  75.30 ± 0.46%
- Ours Ranking(>0.89):  76.14 ± 0.47%

![result](https://i.imgur.com/c4CU2wV.png)

## Predicting Fashionability
Comming soon.
