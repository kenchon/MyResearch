# MyResearch
We propose new method for learning weakly-labeled data based on the [StyleNet](http://hi.cs.waseda.ac.jp/~esimo/publications/SimoSerraCVPR2016.pdf) architecture.

# Codes
- `train.py`: train the model on CPU.
- `load_model.py`: load the model
- `image_sampling.py`: code related to image sampling.
- `stylenet.t7`: StyleNet model which should be converted to PyTorch model.
- `convert_torch.py`: convert Torch7 model to PyTorch.

# Experimental Result
## Classification Task
We evaluate our model by classification task on Hipster Wars dataset which includes 5 class and 1900 images.

### Accuracy Result
- StyleNet Ranking: 74.6%
- StyleNet Joint: 75.9%
- Ours Ranking: 76.4%

![result](https://i.imgur.com/Sic3wec.png)

## Predicting Fashionability
実験実装中
