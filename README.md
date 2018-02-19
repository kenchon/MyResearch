# MyResearch
[StyleNet](http://hi.cs.waseda.ac.jp/~esimo/publications/SimoSerraCVPR2016.pdf)の手法を改良し，WEBで収集できるweakly-labeledな画像を利用したCNNの新しい学習法について提案する。この手法によって，ファッションスタイルのクラス分類におけるstate-of-the-artの75.6%を上回る87.5%の精度を達成した。

# Codes
- `train.py`: train the model on CPU.
- `load_model.py`: load the model
- `image_sampling.py`: code related to image sampling.
- `stylenet.t7`: learned parameters of StyleNet which needs to be converted to PyTorch model.
- `convert_torch.py`: convert Torch7 model to PyTorch.

# Experimental result
CNN学習の100回ごとに，Hipster Warsデータセットのクラス分類を行う。100回ごとのクラス分類の精度は次のように推移した。実験は[論文](http://hi.cs.waseda.ac.jp/~esimo/publications/SimoSerraCVPR2016.pdf)と同じ条件のもとで行った。
![result](https://lh4.googleusercontent.com/opLfLd26jE5OW21Qyb8RWn7KH8bdsr1CdP_PDv7TB-E-SiXIp0-_Jvr8x7Ei8O-VOkubyleLr-ZKns_9PAaB=w1334-h953-rw)
