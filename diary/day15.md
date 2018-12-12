### 181212(Wed)    
p.201 - p.216
chapter6.py : L358 - L514
##### *Remember me*  
シーケンス（= データセット中でランクi（1から始まる）の単語がインデックスiを持つ単語インデックスのリスト）  
-> 文の順番にインデックスがついてる

```python
#Kerasを使った単語レベルでのone-hotエンコーディング
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

#出現頻度の高い1000この単語だけを処理する設定
tokenizer = Tokenizer(num_words=1000)

#単語にインデックスをつける
tokenizer.fit_on_texts(samples)

#文字列を整数インデックスのリストに変換
sequences = tokenizer.texts_to_sequences(samples)
sequences #[[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]

#文字列をone-hotエンコーディングすることも可能
kone_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
one_hot_results.shape #(2, 1000)
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

リカレントニューラルネットワーク  
-> 文章を流れるよう目で追う  
-> 全身的に処理する  
-> 過去の情報から構築され、新しい情報が与えられるたびに更新される

ループと状態をもつ

RNNの入力の形状
```python
(timesteps, input_features)
```

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

#最終的な出力は形状が(timesteps, output_features)の2次元テンソル
final_output_sequence = np.stack(successive_outputs, axis=0)
#successive_outputsのままだと配列がいっぱい入ったリストだったが、それを全体で配列にした
```
SimpleRNN  
```python
#最後のtimestepsのみを出力
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
```python
#完全なシーケンスを返す
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
SimpleRNNでは勾配消失問題がおこる  
-> LSTM層  
-> 超短期記憶  
シーケンスからの情報は何らかの時点でベルトコンベアーに載せられて先のtimestepsへ送られ、必要になったときにそのままの状態でベルトコンベアーから降ろされる  
-> 過去の情報をあとから再注入できるようにすることで購買消失問題に対処

```python
#LSTM層
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

```
