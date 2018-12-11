### 181211(Tue)    
p.195 - p.199  
chapter6.py : L130 - L174
##### *Remember me*  
```python
#埋め込み層(Embedding 層)をインスタンス化
from keras.layers import Embedding

#Embedding層の引数は少なくとも2つ:
#   有効なトークンの数 : この場合は1000(1 + 単語のインデックスの最大値)
#   埋め込みの次元の数 : この場合は64
embedding_layer = Embedding(1000, 64)
#   これに(サンプルの数, サンプル1つあたりの長さ) の形状のテンソルを入力
#   (サンプルの数, サンプル1つあたりの長さ, 埋め込みの次元数) の形状のテンソルが出力される
```
```python
#IMDBデータを読み込み
from keras.datasets import imdb
from keras import preprocessing

#特徴量として考慮する単語の数
max_features = 10000

#max_features個の最も出現頻度の高い単語のうち
#この数の単語を残してテクストカット
max_len = 20

#データを整数のリストとして読み込む
(x_train, y_train), (x_test, y_test) =\
    imdb.load_data(num_words=max_features)

x_train.shape #(25000,)

#整数のリストを形状が(samples, max_len)の整数型の2次元テンソルに変換
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_train.shape #(25000, 20)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

```
