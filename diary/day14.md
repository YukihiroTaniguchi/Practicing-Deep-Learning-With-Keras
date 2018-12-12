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
投入データのトークン化  
```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

max_len = 100               # 映画レビューを100ワードでカット
training_samples = 200      # 200個のサンプルで訓練
validation_samples = 10000  # 10000個のサンプルで検証
max_words = 10000           # データセットの最初から10000ワードのみを考慮

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index)) #Found 88582 unique tokens.

data = pad_sequences(sequences, maxlen=max_len)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape) #Shape of data tensor: (25000, 100)
print('Sahpe of label tensor:', labels.shape) #Sahpe of label tensor: (25000,)

#データを訓練データセットと検証データセットに分割 :
#ただし、サンプルが順番に並んでいる(否定的なレビューのあとに肯定的なレビューが
#配置されている)状態のデータを使用するため、最初にデータをシャッフル
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = data[training_samples: training_samples + validation_samples]
```

Gloveの単語埋め込みを使用してEmbedding層の単語埋め込み行列を準備する

```python
#Glove の単語埋め込みファイルを解析

glove_dir = '/Users/yukihiro/Documents/Practicing-Deep-Learning-With-Keras/data/glove.6B'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index)) #Found 400000 word vectors.
```
```python
#Gloveの単語埋め込み行列の準備
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # 埋め込みインデックスで見つからない単語は0で埋める
            # 自動的に0で埋められる
            embedding_matrix[i] = embedding_vector
```
