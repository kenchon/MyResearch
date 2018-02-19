# MyResearch
[StyleNet](http://hi.cs.waseda.ac.jp/~esimo/publications/SimoSerraCVPR2016.pdf)の手法を改良し，WEBで収集できるweakly-labeledな画像を利用したCNNの新しい学習法について提案する。この手法によって，ファッションスタイルのクラス分類におけるstate-of-the-artの75.6%を上回る87.5%の精度を達成した。

# codes
- `train.py`: train the model on CPU.
- `load_model.py`: load the model([StyleNet](http://hi.cs.waseda.ac.jp/~esimo/publications/SimoSerraCVPR2016.pdf)).
- `image_sampling.py`: code related to image sampling.
- `stylenet.t7`: learned parameters of StyleNet which needs to be converted to PyTorch model.
- `convert_torch.py`: convert Torch7 model to PyTorch.

# experimental result
CNN学習の100回ごとに，Hipster Warsデータセットのクラス分類を行う。100回ごとのクラス分類の精度は次のように推移した。実験は[論文](http://hi.cs.waseda.ac.jp/~esimo/publications/SimoSerraCVPR2016.pdf)と同じ条件のもとで行った。
![result](https://drive.google.com/file/d/1rtcPF4NTbjPyIg9YiqTOTD1ZPSZUZ83G/view?usp=sharing)
