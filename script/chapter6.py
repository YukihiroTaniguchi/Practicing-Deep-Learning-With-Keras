#6.1.1 単語と文字の one-hot エンコーディング
import numpy as np

#初期データ : サンプルごとにエントリが1つ含まれている
#(ここではサンプルは単なる1つの文章だが、文書全体でもよい)
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

#単語レベルでの単純なone-hotエンコーディング
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
results.shape





#文字レベルでの単純なone-hotエンコーディング
import string

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable #すべて印字可能なASCII文字
token_index = dict(zip(characters, range(1, len(characters) + 1)))

token_index
max_length = 50

results = np.zeros((len(samples),
                    max_length,
                    max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, character in enumerate(sample[:max_length]):
        index = token_index.get(character)
        results[i, j, index] = 1.

results.shape
results

#Kerasを使った単語レベルでのone-hotエンコーディング
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

#出現頻度の高い1000この単語だけを処理する設定
tokenizer = Tokenizer(num_words=1000)

#単語にインデックスをつける
tokenizer.fit_on_texts(samples)
#文字列を整数インデックスのリストに変換
sequences = tokenizer.texts_to_sequences(samples)
sequences
#文字列をone-hotエンコーディングすることも可能
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
one_hot_results.shape

print(len(tokenizer.word_index))
#ハッシュトリックを用いた単語レベルの単純なone-hot エンコーディング
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

results.shape

#6.1.2 単語埋め込み

#埋め込み層(Embedding 層)をインスタンス化
from keras.layers import Embedding

#Embedding層の引数は少なくとも2つ:
#   有効なトークンの数 : この場合は1000(1 + 単語のインデックスの最大値)
#   埋め込みの次元の数 : この場合は64
embedding_layer = Embedding(1000, 64)

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

x_train.shape
x_train
#整数のリストを形状が(samples, max_len)の整数型の2次元テンソルに変換
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_train.shape
x_train


x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

model = Sequential()

#あとから埋め込み入力を平坦化できるよう、
#Embedding層に入力の長さとしてmax_lenを指定
#Embedding層のあと、活性化の形状は(samples, max_len, 8)になる
model.add(Embedding(10000, 8, input_length=max_len))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)


#6.1.3 テキストのトークン化から単語埋め込みまで
import os

# IMDb データセットが置かれているディレクトリ
pwd
imdb_dir = '/Users/yukihiro/Documents/Practicing-Deep-Learning-With-Keras/data/aclImdb'
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
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=max_len)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Sahpe of label tensor:', labels.shape)

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
y_val = labels[training_samples: training_samples + validation_samples]

pwd

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

print('Found %s word vectors.' % len(embeddings_index))

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

#モデルの定義
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#準備した単語埋め込みをEmbeddingに読み込む
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

#コンパイル、訓練、評価
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

model.save_weights('pre_trained_glove_model.h5')

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#正解率をプロット
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

#損失値をプロット
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

#学習済みの単語埋め込みを使用せずに同じモデルを訓練
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#正解率をプロット
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

#損失値をプロット
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

#テストデータセットのデータをトークン化
test_dir = os.path.join(imdb_dir, 'test')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=max_len)
y_test = np.asarray(labels)

#モデルをテストデータセットで評価
model.load_weights('others/pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)

#6.2 リカレントニューラルネットワーク
#RNNの擬似コード
state_t = 0                         #時間tでの状態
for input_i in input_sequence:      #シーケンスの要素をループで処理
    output_t = f(input_t, state_t)  #この1つ前の出力が
    state_t = output_t              #次のイテレーションの状態になる

#より詳細なRNNの疑似コード
state_t = 0
for input_t in input_sequence:
    output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
    state_t = output_t

#単純なNumPyによるRNNの疑似コード
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

#6.2.1 Keras でのリカレント層
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences = True))
model.summary()

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.summary()

#IMDbデータの前処理
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000 #特徴量として考慮する単語の数
max_len = 500   #この数の単語を残してテキストカット
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = \
    imdb.load_data(num_words=max_features)

print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=max_len)
input_test = sequence.pad_sequences(input_test, maxlen=max_len)

print('input_train shape:', input_train.shape)
print('input_test shape', input_test.shape)

#Embedding層 とSimpleRNN層を使ってモデルを訓練
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN,Dense

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(input_train, y_train,
                    epochs=10, batch_size=128, validation_split=0.2)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_acc, 'b', label='Validation loss')
plt.title('Training an validation loss')
plt.legend()

#6.2.2 LSTM層とGRU層

#LSTM アーキテクチャの擬似コード
output_t = activation(dot(state_t, Uo) + dot(input_t, Wo)
                                       + dot(C_t, Vo) + bo)

i_t = activation(dot(state_t, Ui) + dot(input_t, Wi) + bi)
f_t = activation(dot(state_t, Uf) + dot(input_t, Wf) + bf)
k_t = activation(dot(state_t, Uk) + dot(input_t, Wk) + bk)

#新しいキャリー状態
c_t+1 = i_t * k_t + c_t * f_t

#6.2.3 KerasでのLSTMの具体的な例
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
