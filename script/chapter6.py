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

#文字列をone-hotエンコーディングすることも可能
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
one_hot_results.shape

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

埋め込み層(Embedding 層)をインスタンス化
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