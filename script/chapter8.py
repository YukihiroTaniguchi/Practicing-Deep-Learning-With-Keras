import numpy as np

# 8.1.3 サンプリング戦略の重要性

# 異なる温度での確率分布の再荷重
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


# 8.1.4 LSTMによる文字レベルのテキスト生成
import keras
import numpy as np

path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length:', len(text))

# 文字のシーケンスのベクトル化
maxlen = 60     # 60文字のシーケンス
step = 3        # 3文字お気に新しいシーケンスをサンプリング
sentences = []  # 抽出されたシーケンスを保持
next_chars = [] # 目的地(次に来る文字)を保持

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print('Number of sequences:', len(sentences))

# コーパスの一意な文字のリスト
chars = sorted(list(set(text)))
print('Unique characters:', len(chars))

print(chars)

# これらの文字をリストcharsのインデックスにマッピングする
char_indices = dict((char, chars.index(char)) for char in chars)

print(char_indices)

print('Vectorization...')

# one-hotエンコーディングを適用して文字を二値の配列に格納
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# 次の文字を予測する単相LSTMモデル
from keras import layers

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

# モデルのコンパイル設定
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy')

# モデルの予測に基づいて次の文字をサンプリングする関数
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# テキスト生成ループ
import random
import sys

# モデルを60エポックで訓練
for epoch in range(1, 60):
    print('epoch', epoch)

    # 1エポックでデータを学習
    model.fit(x, y, batch_size=128, epochs=1)

    # テキストシードをランダムに選択
    start_index = random.randint(0, len(text) - maxlen -1)
    gererated_text = text[start_index: start_index + maxlen]
    print('--- Gererating with sedd: "' + gererated_text + '"')

    # ある範囲内の異なるサンプリング温度を試してみる
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(gererated_text)

        # 400文字を生成
        for i in range(400):

            # これまでに生成された文字にone-hotエンコーディングを適用
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(gererated_text):
                sampled[0, t, char_indices[char]] = 1.

            # 次の文字をサンプリング
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            gererated_text += next_char
            gererated_text = gererated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
