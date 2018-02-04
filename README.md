# MyResearch
This repository is for my own reserch and related to feature extraction from images.
This research is based on this [paper](http://hi.cs.waseda.ac.jp/~esimo/publications/SimoSerraCVPR2016.pdf).

I use the model learned on the Fashion144K and retrain the model on ours loss function.

- `train_gpu.py`: train the model on GPU but not yet debugged.
- `train.py`: train the model on CPU.
- `load_model.py`: load the model([StyleNet](http://hi.cs.waseda.ac.jp/~esimo/publications/SimoSerraCVPR2016.pdf)).
- `image_sampling.py`: code related to image sampling.
- `stylenet.t7`: learned parameters of StyleNet which needs to be converted to PyTorch model.
- `convert_torch.py`: convert Torch7 model to PyTorch.
- `loss_result.txt`: transision of loss
