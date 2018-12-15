#### chapter6 まとめ  
##### テキストデータを有益な表現にする前処理
テキストのベクトル化
複数の方法
1. テキストを単語に分割、各単語をベクトルに変換する
2. テキストを文字に分割し、各文字をベクトルに変換する
3. Nグラムの単語または文字をちゅうしゅつし、各Nグラムをベクトルに変換する

->これらの単位をトークンと呼び、トークン化すると呼ぶ
トークン化
1. one-hot エンコーディング
2. トークン埋め込み(単語埋め込み)

単語レベルでの単純なone-hotエンコーディング
```python
token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            #一意な単語にそれぞれ一意なインデックスを割当する
            #インデックス0をどの単語にも割り当てないことに注意
            token_index[word] = len(token_index) + 1

print(token_index) #{'The': 1, 'cat': 2, 'sat': 3, 'on': 4, 'the': 5, 'mat.': 6, 'dog': 7, 'ate': 8, 'my': 9, 'homework.': 10}

#サンプルをベクトル化 : サンプルごとに最初のmax_length個の単語丈を考慮
max_length = 10

#結果の格納場所
results = np.zeros((len(samples),
                    max_length,
                    max(token_index.values()) + 1))

for i, sample in enumerate(samples):
   for j, word in list(enumerate(sample.split()))[:max_length]:
       index = token_index.get(word)
       results[i, j, index] = 1.
results.shape #(2, 10, 11)
```

#文字レベルでの単純なone-hotエンコーディング
```python
import string

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable #すべて印字可能なASCII文字
token_index = dict(zip(characters, range(1, len(characters) + 1)))

token_index
# {'0': 1,
#  '1': 2,
#  '2': 3,
#  '3': 4,
#  '4': 5,
#  '5': 6,
#  '6': 7,
#  '7': 8,
#  '8': 9,
#  '9': 10,
#  'a': 11,
#  'b': 12,
#  'c': 13,
#  'd': 14,
#  'e': 15,
#  'f': 16,
#  'g': 17,
#  'h': 18,
#  'i': 19,
#  'j': 20,
#  'k': 21,
#  'l': 22,
#  'm': 23,
#  'n': 24,
#  'o': 25,
#  'p': 26,
#  'q': 27,
#  'r': 28,
#  's': 29,
#  't': 30,
#  'u': 31,
#  'v': 32,
#  'w': 33,
#  'x': 34,
#  'y': 35,
#  'z': 36,
#  'A': 37,
#  'B': 38,
#  'C': 39,
#  'D': 40,
#  'E': 41,
#  'F': 42,
#  'G': 43,
#  'H': 44,
#  'I': 45,
#  'J': 46,
#  'K': 47,
#  'L': 48,
#  'M': 49,
#  'N': 50,
#  'O': 51,
#  'P': 52,
#  'Q': 53,
#  'R': 54,
#  'S': 55,
#  'T': 56,
#  'U': 57,
#  'V': 58,
#  'W': 59,
#  'X': 60,
#  'Y': 61,
#  'Z': 62,
#  '!': 63,
#  '"': 64,
#  '#': 65,
#  '$': 66,
#  '%': 67,
#  '&': 68,
#  "'": 69,
#  '(': 70,
#  ')': 71,
#  '*': 72,
#  '+': 73,
#  ',': 74,
#  '-': 75,
#  '.': 76,
#  '/': 77,
#  ':': 78,
#  ';': 79,
#  '<': 80,
#  '=': 81,
#  '>': 82,
#  '?': 83,
#  '@': 84,
#  '[': 85,
#  '\\': 86,
#  ']': 87,
#  '^': 88,
#  '_': 89,
#  '`': 90,
#  '{': 91,
#  '|': 92,
#  '}': 93,
#  '~': 94,
#  ' ': 95,
#  '\t': 96,
#  '\n': 97,
#  '\r': 98,
#  '\x0b': 99,
#  '\x0c': 100}

max_length = 50

results = np.zeros((len(samples),
                    max_length,
                    max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, character in enumerate(sample[:max_length]):
        index = token_index.get(character)
        results[i, j, index] = 1.

results.shape #(2, 50, 101)
```

Kerasを使った単語レベルでのone-hotエンコーディング
```python
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

#出現頻度の高い1000個の単語だけを処理する設定
tokenizer = Tokenizer(num_words=1000)

#単語にインデックスをつける
tokenizer.fit_on_texts(samples)

#文字列を整数インデックスのリストに変換
sequences = tokenizer.texts_to_sequences(samples)
print(sequences) #[[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]

#文字列をone-hotエンコーディングすることも可能
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
one_hot_results.shape #(2, 1000)
print(one_hot_results)
#[[0. 1. 1. ... 0. 0. 0.]
# [0. 1. 0. ... 0. 0. 0.]]
```

one-hot ハッシュトリック  
-> 語彙に含まれている一名トークンの数が多すぎる場合に有効  
-> ディクショナリで参照する代わりに固定サイズのベクトルにハッシュ化
-> ハッシュ衝突に注意
ハッシュトリックを用いた単語レベルの単純なone-hot エンコーディング
```python
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

#単語をサイズが1000のベクトルに格納
#単語の数がこれに近いまたはそれ以上の場合
#ハッシュ衝突が頻発する
dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        #単語をハッシュ化し、0-1000のランダムな整数に変換
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.

results.shape #(2, 10, 1000)
```

トークン埋め込み(単語埋め込み)  
one-hot エンコーディングよりも遥かに少ない次元数でより多くの情報を格納する

単語埋め込みを取得する方法
1. メインのタスク(文書分類や感情予測など)と同時に単語埋め込みを学習する
2. 別の機械学習タスクを使って計算された単語埋め込みをモデルに読み込む  
-> 学習済みの単語埋め込み

単語ベクトル同士の幾何学的な関係は、それらの単語の意味的な関係を反映したものでなければならない  
-> 人間の言語を幾何学的な空間へマッピングする  
意味合いが関連している : 距離的に近い  
-> イヌ科からネコ科へのベクトル : 位置ベクトルが違っても向きが同じ

意味的な関係の重要性はタスクごとに異なっている  
-> 新しいタスクごとに新しい埋め込み空間を学習することが理にかなっている

Kerasの埋め込み層(Embedding層)  
整数のインデックスを密ベクトルにマッピングするディクショナリ

埋め込み層(Embedding 層)をインスタンス化
```python
from keras.layers import Embedding

#Embedding層の引数は少なくとも2つ:
#   有効なトークンの数 : この場合は1000(1 + 単語のインデックスの最大値)
#   埋め込みの次元の数 : この場合は64
embedding_layer = Embedding(1000, 64)
```

入力の形状  
```python
(samples, sequence_length)
 (32, 10) #長さが10の32個のシーケンスからなるバッチ
```  
-> シーケンスはすべて同じ長さでなければならない  

出力の形状
```python
(samples, sequence_length, embedding_dimensinality)
```
-> RNN、1次元のCNNで処理できる

入力シーケンスを揃える
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

len(x_train) #25000
len(x_train[0]) #218

#整数のリストを形状が(samples, max_len)の整数型の2次元テンソルに変換
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_train.shape #(25000, 20)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
x_train.shape #(25000, 20)
```

Embedding層と分類器を使用
```python
from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
# あとから埋め込み入力を平坦化できるよう
# Embedding層に入力の長さとしてmax_lenを指定
# Embedding層の後、活性化の形状は(samples, max_len, 8)になる
model.add(Embedding(10000, 8, input_length=max_len))

# 埋め込み層の後、(samples, maxl_len, 8)3次元テンソルの形状を
# (samples, max_len * 8)の2次元テンソルに変換する
model.add(Flatten())

# 最後に分類器(sigmoidの二値分類)を追加
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)
```

学習済みの単語埋め込みを使用  
-> 利用可能な訓練データが少ないために、  
手持ちのデータだけではタスクに適した語彙の埋め込みを学習できないことがある  
異なる問題で学習された特徴量を再利用するのが得策  

Glove の単語埋め込みファイルを解析(ディクショナリにする)
```python

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
Gloveの単語埋め込み行列の準備(ディクショナリを行列に変換)
```python
max_words = 10000
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
学習済みの特徴量の埋め込みをモデルに読み込む
```python
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#準備した単語埋め込みをEmbeddingに読み込む
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
#せっかく覚えた重みを忘れないように凍結する
```


シーケンス  
（= データセット中でランクi（1から始まる）の単語がインデックスiを持つ単語インデックスのリスト）  
-> 単語の順番にインデックスがついてる  
-> インデックス0はプレースホルダのため何も表さない

参考  
※IMDbデータセットをlabelsリストとtextsリストにまとめる
```python
import os
imdb_dir= \
'/Users/yukihiro/Documents/practice/Practicing-Deep-Learning-With-Keras/data/aclImdb'
# IMDbデータセットが置かれているディレクトリ
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

```


labels、textsリスト化した投入データのトークン化  
```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

max_len = 100               # 映画レビューを100ワードでカット
training_samples = 200      # 200個のサンプルで訓練
validation_samples = 10000  # 10000個のサンプルで検証
max_words = 10000           # データセットの最初から10000ワードのみを考慮

#出現頻度の高い1000この単語だけを処理する設定
tokenizer = Tokenizer(num_words=max_words)

#単語にインデックスをつける
tokenizer.fit_on_texts(texts)

#単語を整数インデックスのリストに変換
sequences = tokenizer.texts_to_sequences(texts)
sequences #[[1, 2, 3, 4, 1, 5...], [1, 6, 7, 8, 9...]...]

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index)) #Found 88582 unique tokens.

#シーケンスを同じ長さになるように詰めるor 伸ばす and 配列に変換
data = pad_sequences(sequences, maxlen=max_len)

#配列に変換
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape) #Shape of data tensor: (25000, 100)
print('Sahpe of label tensor:', labels.shape) #Sahpe of label tensor: (25000,)

#データを訓練データセットと検証データセットに分割 :
#ただし、サンプルが順番に並んでいる(否定的なレビューのあとに肯定的なレビューが
#配置されている)状態のデータを使用するため、最初にデータをシャッフル
indices = np.arange(data.shape[0]) #indices -> indexの複数形
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = data[training_samples: training_samples + validation_samples]
```



##### リカレントニューラルネットワーク(RNN)の操作

リカレントニューラルネットワーク  
-> 文章を流れるよう目で追う  
-> 全身的に処理する  
-> 過去の情報から構築され、新しい情報が与えられるたびに更新される  
-> ループと状態をもつ  

RNNの疑似コード
```python
state_t = 0
for input_t in input_sequence:
    output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
    state_t = output_t
```
単純なNumPyによるRNNの疑似コード
```python
import numpy as np

timesteps = 100         #入力シーケンスの timesteps の数
input_features = 32     #入力特徴空間の次元の数
output_features = 64   #出力特徴空間の次元の数

#入力データ : ランダムにノイズを挿入
inputs = np.random.random((timesteps, input_features))

#初期状態 : すべての0のベクトル
state_t = np.zeros((output_features))

#ランダムな重み行列を作成
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features))

successive_outputs = []

#input_tは形状が(input_features, )のベクトル
for input_t in inputs:
    #入力と現在の状態(1つ前の出力)を結合して現在の出力を取得
    #活性化関数tanh
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    #この出力をリストに格納
    successive_outputs.append(output_t)
    #次の時間刻みのためにRNNの状態を更新
    state_t = output_t

#最終的な出力は形状が(timesteps, output_features)
#の2次元テンソル
final_output_sequence = np.stack(successive_outputs, axis=0)
#successive_outputsのままだと
#配列がいっぱい入ったリストだったが、
#それを全体で配列にした(np.stackでならした)
```
多くの場合上のような完全なシーケンスは不要  
-> 最後の出力のみで良い

Keras のリカレント層  
SimpleRNN  
最後のtimestepsのみ出力
```python
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, None, 32)          320000    
_________________________________________________________________
simple_rnn_3 (SimpleRNN)     (None, 32)                2080      
=================================================================
Total params: 322,080
Trainable params: 322,080
Non-trainable params: 0
_________________________________________________________________
```
完全なシーケンスを返す
```python
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences = True))
model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_4 (Embedding)      (None, None, 32)          320000    
_________________________________________________________________
simple_rnn_4 (SimpleRNN)     (None, None, 32)          2080      
=================================================================
Total params: 322,080
Trainable params: 322,080
Non-trainable params: 0
_________________________________________________________________
```
Embedding層の出力
```python
(samples, sequence_length, embedding_dimensinality)
```
↓  
SimpleRNNの入力
```python
(batch_size, timesteps, input_features)
```

SimpleRNNより高度なリカレント層

SimpleRNNでは勾配消失問題がおこる  
-> LSTM層  
-> 超短期記憶  
シーケンスからの情報は何らかの時点でベルトコンベアーに載せられて先のtimestepsへ送られ、必要になったときにそのままの状態でベルトコンベアーから降ろされる  
-> 過去の情報をあとから再注入できるようにすることで購買消失問題に対処

LSTM層
```python
from keras.layers import LSTM
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
                    #訓練データの後ろ20％が検証データになる
```
```python
float_data.shape # (420551, 14)
```

時系列サンプルとそれらのターゲットを生成するデータジェネレーター  
```python

data : もとの浮動小数点数型のデータからなる配列  
lookback : 入力データのtimestepsをいくつさかのぼるか  
delay : Targetのtimestepsをいくつ進める化  
min_index, max_index : 抽出するtimestepsの上限と下限を表すdata配列のインデックス。
データの一部を検証とテストのためにとっておくのに役立つ  
どっからどこまでかを表すことになる
shuffle : サンプルをシャッフルするのか、それとも時間の順序で抽出するのか  
batch_size : バッチひとつあたりのサンプル数  
step : データをサンプリングする時の期間(単位はtimesteps)データポイントを1時間ごとに1つ抽出するために6に設定

def generator(data, lookback, delay, min_index, max_index,
            shuffle=False, batch_size=128, step=6):
  if max_index is None:
      max_index = len(data) - delay - 1
  i = min_index + lookback
  while 1:
      if shuffle:
          rows = np.random.randint(min_index + lookback, max_index,
                                   size=batch_size)
      else:
          if i + batch_size >= max_index:
              i = min_index + lookback
          rows = np.arange(i, min(i + batch_size, max_index))
          i += len(rows)

      samples = np.zeros((len(rows), #バッチひとつあたりのサンプルの数
                          lookback // step, #240時間 -> 5日間
                          data.shape[-1])) #特徴量の数
      targets = np.zeros((len(rows),))
      for j, row in enumerate(rows):
          indices = range(rows[j] - lookback, rows[j], step)
          samples[j] = data[indices]
          targets[j] = data[rows[j] + delay][1]
    yield samples, targets

```
訓練、検証、テストジェネレータの準備
```python

lookback = 1440
step = 6
delay = 144
batch_size = 128

#訓練ジェネレータ
train_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=0,
                     max_index=200000,
                     shuffle=True,
                     step=step,
                     batch_size=batch_size)

#検証ジェネレータ
val_gen = generator(float_data,
                   lookback=lookback,
                   delay=delay,
                   min_index=200001,
                   max_index=300000,
                   step=step,
                   batch_size=batch_size)

#テストジェネレータ
test_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=300001,
                    max_index=None,
                    step=step,
                    batch_size=batch_size)

#検証データセット全体を調べるためにval_genから抽出するtimestepsの数
val_steps = (300000 - 200001 - lookback) // batch_size

#テストデータセット全体を調べるためにtest_genから抽出するtimestepsの数
test_steps = (len(float_data) - 300001 - lookback) // batch_size
```
常識的なベースラインを設定する  
Aクラスのインスタンス : 90%  
Bクラスのインスタンス : 10%  
-> 常識的なアプローチ 常にAクラスを予測する  
-> 正解率90%
これを超えるようにする

RNNなどの複雑で計算負荷の高い機械学習モデルを調べる再には、単純で計算不可の低いモデルを試してみる
-> 計算負荷の低い全結合モデル
```python
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step,
                                      float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
# -> 回帰問題では最後のDense層に活性化関数を指定しないのが一般的

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
```

GRU層  
-> LSTM層よりも実行コストが掛からない  
-> LSTM層ほど表現力がない
-> 遠い過去よりも最近のことを覚えるほうが得意

```python
#GRUベースのモデルの訓練と評価
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))#特徴量の数
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
```

リカレントドロップアウト  
-> timestepsごとにドロップアウトマスクをランダムに変化させるのではなく  
-> すべてのtimestepsごとに同じドロップアウトマスクを適用する  
-> すべてのtimestepsで同じドロップアウトマスクを使用すると、ネットワークが時間の流れに沿って学習誤差を正しく伝播できる

リカレント層のスタッキング  
-> 過学習を抑える基本的な手続きが取られているときのみ  
-> ネットワークのキャパシティを増やしてみる
-> 層のユニットを増やすか、層を更に増やす
-> Kerasのリカレント層をスタックとして積み上げるには、
-> すべての中間層(最後意外)が完全な出力シーケンスを返さなけらばならない


```python
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1, #ドロップアウトマスク
                     recurrent_dropout=0.5, #リカレントドロップアウトマスク
                     return_sequences=True, #スタッキングのための完全な出力をTrue
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
                     #最後だから完全な出力はいらない
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)
```

双方向RNN  
-> 一方向のRNN では見落とされるかもしれないパターンを補足することができる  
-> たとえ違っていても有益である表現には常に利用価値がある  
-> 自然言語処理(NLP)の問題において有益  
-> 最近のデータのほうがはるかに情報利得の高いシーケンスデータでは微妙

GRUベースの双方向RNNの訓練と評価  
Bidirectional層  
最初の引数としてリカレント層を受け取る
```python
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Bidirectional(layers.GRU(32,
                               input_shape=(None, float_data.shape[-1]))))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)
```

リカレントアテンション  
シーケンスマスキング

##### 1次元の畳込みニューラルネットワーク(CNN)を使ったシーケンス処理

CNNは特定のシーケンス処理問題においてRNNの好敵手となる  
さらにRNNより計算コストが掛からない

1次元の畳込み  
-> 文字シーケンスを処理する場合  
-> ウインドウのサイズとして5を使用している場合  
-> 長さが5以下の単語または単語の一部を学習できる  
-> 移動不変
-> 文字レベルの1次元CNNは単語の形態を学習できる

Embedding層の出力
```python
(samples, sequence_length, embedding_dimensinality)
```
↓  
1次元CNNの入力
```python
(samples, time, features)
```
1次元CNNと2次元CNNの違い  
1次元CNN はより大きな畳込みウィンドウを使用することが可能

```python
#IMDbデータセットでの単純な1次元CNNの訓練と評価
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu')) #7 -> ウィンドウの大きさ
model.add(layers.MaxPooling1D(5)) #ウィンドウの大きさ
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D()) #グローバルプーリング層
model.add(layers.Dense(1))

model.summary()
model.compile(optimizer=RMSprop(lr=1e-4),
              loss = 'binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
```

CNNとRNNを組み合わせて長いシーケンスを処理する  
CNNの軽快さをRNNの順序への敏感さと組み合わせる  
-> 1次元CNNをRNNの前処理ステップとして使用すること  
-> RNNで処理するのが現実的ではないほど長いシーケンスを扱っている場合  
-> 逆にかなりさかのぼって調べたり、分解能の高い時系列データを調べたりすることが可能

分解能の高いデータジェネレータの準備
```python
lookback = 720
step = 3
delay = 144

#訓練ジェネレータ
train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step)

val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step)

test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step)

#検証データセット全体を調べるためにval_genから抽出するtimestepsの数
val_steps = (300000 - 200001 - lookback) // 128

#テストデータセット全体を調べるためにtest_genから抽出するtimestepsの数
test_steps = (len(float_data) - 300001 - lookback) // 128

```
1次元畳み込みベースとGRU層で構成されたモデル
```python
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',
                        input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.summary()
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
```
