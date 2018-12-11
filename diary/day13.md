### 181210(Mon)    
p.188 - p.196  
chapter6.py : L1 - L129
##### *Remember me*  
テキストデータの操作  
テキストのベクトル化  
1. テキストを単語に分割、各単語をベクトルに変換する
2. テキストを文字に分割し、各文字をベクトルに変換する
3. Nグラムの単語または文字をちゅうしゅつし、各Nグラムをベクトルに変換する

->これらの単位をトークンと呼び、トークン化すると呼ぶ
トークン化
1. one-hot エンコーディング
2. トークン埋め込み(単語埋め込み)

one-hot エンコーディング
```python
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
results.shape #(2, 10, 11)
```
```python
#文字レベルでの単純なone-hotエンコーディング
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

#文字列をone-hotエンコーディングすることも可能
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
one_hot_results.shape #(2, 1000)
```
```python
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

```python
#埋め込み層(Embedding 層)をインスタンス化
from keras.layers import Embedding

#Embedding層の引数は少なくとも2つ:
#   有効なトークンの数 : この場合は1000(1 + 単語のインデックスの最大値)
#   埋め込みの次元の数 : この場合は64
embedding_layer = Embedding(1000, 64)
```

それぞれのシーケンスはすべて同じ長さでなければならない  
