# 7.1.1 速習 : Keras Functional API
from keras import Input, layers

# テンソル
input_tensor = Input(shape=(32,))

# 層は関数
dense = layers.Dense(32, activation='relu')

# テンソルで呼び出された層はテンソルを返す
output_tensor = dense(input_tensor)

from keras.models import Sequential, Model

# すでにおなじみのSequentialモデル
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64, )))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

# Functional API で上のSequentialモデルに相当するもの
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

# Modelクラスは入力テンソルと出力テンソルをモデルに変換する
model = Model(input_tensor, output_tensor)

# このモデルのアーキテクチャを確認
model.summary()

# モデルをコンパイル
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 訓練に使用するダミーのNumPyデータを生成
import numpy as np
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

# モデルを10エポックで訓練
model.fit(x_train, y_train, epochs=10, batch_size=128)

# モデルを評価
score = model.evaluate(x_train, y_train)

# 7.1.2 他入力モデル
# 2つの入力を持つ質問応答モデルのFunctional API 実装
# 入力1 : 質問に答えるための情報を提供するテキスト
# 入力2 : 質問のテキスト
# 出力1 : 答え
from keras.models import Model
from keras import layers
from keras import Input

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

# テキスト入力は整数の可変長のシーケンス
# なお、ひつようであれば、入力に名前をつけることもできる
text_input = Input(shape=(None,), dtype='int32', name='text')

#入力をサイズが64のベクトルシーケンスに埋め込む
embedded_text = layers.Embedding(
    text_vocabulary_size, 64)(text_input)

# LSTMを通じてこれらのベクトルを単一のベクトルにエンコード
encorded_text = layers.LSTM(32)(embedded_text)

#質問入力でも(異なる層のインスタンを使って)同じプロセスを繰り返す
question_input = Input(shape=(None, ), dtype='int32', name='question')
embedded_question = layers.Embedding(
    question_vocablulary_size, 32)(question_input)
encorded_question = layers.LSTM(16)(embedded_question)

# エンコードされたテキストと質問を連結
concatenated = layers.concatenate([encorded_text, encorded_question],
                                  axis=-1)

# ソフトマックス分類器を追加
answer = layers.Dense(
    answer_vocabulary_size, activation='softmax')(concatenated)

# モデルをインスタンス化するときには、2つの入力と1つの出力を指定
model = Model([text_input, question_input], answer)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

# 他入力モデルへのデータの供給
import numpy as np
num_samples = 1000
max_length = 100

# ダミーのNumPyデータを生成
text = np.random.randint(1, text_vacabulary_size,
                         size=(num_samples, max_length))

question = np.random.randint(1, question_vocablulary_size,
                             size=(num_samples, max_length))

# 答えに(整数ではなく)one-hotエンコーディングを適用
answers = np.zeros(shape=(num_samples, answer_vocabulary_size))
indices = np.random.randint(0, answer_vocabulary_size, size=num_samples)
for i, x in enumerate(answers):
    x[indices[i]] = 1
