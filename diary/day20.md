### 181220(Thu)    
p.284 - p.
chapter8.py : L1 - L121
##### *Remember me*  
ジェネレーティブリカレントネットワーク  
-> 文章の生成  
-> 抽出されたN個の文字からなる文字列をLSTM層に与えることでN+ 1個の文字を予測させる

貪欲的サンプリング  
-> 最も有力な候補が常に選択される  
-> 意味の通る文章にならないかもしれない

確率的サンプリング  
-> パラメータを導入し、多少ランダムにする  
より興味深い文章を生成する事が可能

文字全部に対してソフトマックス関数で確率分布を出す  
ソフトマックス温度(softmax temperature)  
を変えることでどのくらい意外な選択をさせるかを調整


異なる温度での確率分布の再荷重
```python
# original_distribution は確率値からなる1次元のNumPy配列
# 確率値の総和は1でなければならない
# temperature は出力分布のエントロピーを定量化する係数
def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)

    # 再荷重された元の確率分布を返す
    # 新しい確率分布の総和は1にならない可能性があるため、
    # 新しい分布を取得するために総和で割っている
    return distribution / np.sum(distribution)
```

LSTMによる文字レベルのテキスト生成
```python
import keras
import numpy as np

path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length:', len(text)) # Corpus length: 600893
```

文字のシーケンスのベクトル化
```python
maxlen = 60     # 60文字のシーケンス
step = 3        # 3文字お気に新しいシーケンスをサンプリング
sentences = []  # 抽出されたシーケンスを保持
next_chars = [] # 目的地(次に来る文字)を保持

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print('Number of sequences:', len(sentences)) # Number of sequences: 200278

# コーパスの一意な文字のリスト
chars = sorted(list(set(text)))
print('Unique characters:', len(chars)) # Unique characters: 57

print(chars)
# ['\n', ' ', '!', '"', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ä', 'æ', 'é', 'ë']

# これらの文字をリストcharsのインデックスにマッピングする
char_indices = dict((char, chars.index(char)) for char in chars)

print(char_indices)
# {'\n': 0, ' ': 1, '!': 2, '"': 3, "'": 4, '(': 5, ')': 6, ',': 7, '-': 8, '.': 9, '0': 10, '1': 11, '2': 12, '3': 13, '4': 14, '5': 15, '6': 16, '7': 17, '8': 18, '9': 19, ':': 20, ';': 21, '=': 22, '?': 23, '[': 24, ']': 25, '_': 26, 'a': 27, 'b': 28, 'c': 29, 'd': 30, 'e': 31, 'f': 32, 'g': 33, 'h': 34, 'i': 35, 'j': 36, 'k': 37, 'l': 38, 'm': 39, 'n': 40, 'o': 41, 'p': 42, 'q': 43, 'r': 44, 's': 45, 't': 46, 'u': 47, 'v': 48, 'w': 49, 'x': 50, 'y': 51, 'z': 52, 'ä': 53, 'æ': 54, 'é': 55, 'ë': 56}


print('Vectorization...')

# one-hotエンコーディングを適用して文字を二値の配列に格納
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
```
次の文字を予測する単相LSTMモデル
```python
from keras import layers

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))
```

モデルのコンパイル設定
```python
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy')
```

モデルの予測に基づいて次の文字をサンプリングする関数
```python
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
```

テキスト生成ループ
```python
import random
import sys

# モデルを60エポックで訓練
for epoch in range(1, 60):
    print('epoch', epoch)

    # 1エポックでデータを学習
    model.fix(x, y, batchsize=128, epochs=1)

    # テキストシードをランダムに選択
    start_index = random.randint(0, len(text) - maxlen -1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Gererating with sedd: "' + generated_text + '"')

    # ある範囲内の異なるサンプリング温度を試してみる
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        # 400文字を生成
        for i in range(400):

            # これまでに生成された文字にone-hotエンコーディングを適用
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            # 次の文字をサンプリング
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
```
