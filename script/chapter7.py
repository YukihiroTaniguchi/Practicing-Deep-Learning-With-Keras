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

# 入力1 : 質問に答えるための情報を提供するテキスト
# テキスト入力は整数の可変長のシーケンス
# なお、ひつようであれば、入力に名前をつけることもできる
text_input = Input(shape=(None,), dtype='int32', name='text')

#入力をサイズが64のベクトルシーケンスに埋め込む
embedded_text = layers.Embedding(
    text_vocabulary_size, 64)(text_input)

# LSTMを通じてこれらのベクトルを単一のベクトルにエンコード
encorded_text = layers.LSTM(32)(embedded_text)

# 入力2 : 質問のテキスト
#質問入力でも(異なる層のインスタンを使って)同じプロセスを繰り返す
question_input = Input(shape=(None, ), dtype='int32', name='question')
embedded_question = layers.Embedding(
    question_vocabulary_size, 32)(question_input)
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
text = np.random.randint(1, text_vocabulary_size,
                         size=(num_samples, max_length))

question = np.random.randint(1, question_vocabulary_size,
                             size=(num_samples, max_length))

# 答えに(整数ではなく)one-hotエンコーディングを適用
answers = np.zeros(shape=(num_samples, answer_vocabulary_size))
indices = np.random.randint(0, answer_vocabulary_size, size=num_samples)
for i, x in enumerate(answers):
    x[indices[i]] = 1

# 入力リストを使った適合
model.fit([text, question], answers, epochs=10, batch_size=128)

# 入力ディクショナリを使った適合(入力に名前をつける場合)
model.fit({'text': text, 'question': question}, answers,
         epochs=10, batch_size=128)

# 7.1.3 他出力モデル
# 3つの出力を持つモデルのFunctional API実装
# 入力1 : ソーシャルメディアへの投稿
# 出力1 : 年齢
# 出力2 : 所得
# 出力3 : 性別
from keras import layers
from keras import Input
from keras.models import Model

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(vocabulary_size, 256)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

# 出力層に名前がついていることに注意
# 出力1 : 年齢
age_prediction = layers.Dense(1, name='age')(x)
# 出力3 : 性別
income_prediction = layers.Dense(num_income_groups,
                                 activation='softmax',
                                 name='income')(x)
# 出力3 : 性別
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)
model = Model(posts_input,
              [age_prediction, income_prediction, gender_prediction])

# 他出力モデルのコンパイルオプション(複数の損失)
model.compile(optimizer='rmsprop',
              loss=['mse',
                    'categorical_crossentropy',
                    'binary_crossentropy'])

# 上記と同じ(出力層に名前をつけている場合のみ可能)
model.compile(optimizer='rmsprop',
              loss={'age': 'mse',
                    'income': 'categorical_crossentropy',
                    'gender': 'binary_crossentropy'})

# 多出力モデルのコンパイルオプション(損失の重み付け)
model.compile(optimizer='rmsprop',
              loss=['mse',
                    'categorical_crossentropy',
                    'binary_crossentropy'],
              loss_weights=[0.25, 1., 10.])

# 上記と同じ(出力層に名前をつけている場合に飲み可能)
model.compile(optimizer='rmsprop',
              loss={'age': 'mse',
                    'income': 'categorical_crossentropy',
                    'gender': 'binary_crossentropy'},
              loss_weights={'age': 0.25, 'income': 1, 'gender': 10.})

# 多出力モデルへのデータの供給
# age_targets, income_targets, gender_targetsはNumPy配列と仮定
model.fit(posts, [age_targets, income_targets, gender_targets],
          epochs=10, batch_size=64)

# 上記と同じ(出力層に名前をつけている場合のみ可能)
model.fit(posts, {'age': age_targets,
                  'income': income_targets,
                  'gender': gender_targets},
          epochs=10, batch_size=64)

# 7.1.4 層の有効非巡回グラフ
# Inception モジュール
from keras import layers

# 各分岐のストライドの値は同じ (2) :
# すべての分岐の出力を同じサイズに保って連結可能にするために必要

# 分岐a
branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)

# この分析では、空間畳み込み層でストライドが発生する
# 分岐b
branch_b = layers.Conv2D(128, 1, activation='relu')(x)
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)

# この分岐では、平均値プーリング層でストライドが発生する
# 分岐c
branch_c = layers.AveragePooling2D(3, strides=2)(x)
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)

# 分岐d
branch_d = layers.Cov2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)

# モジュールの出力を取得するために分岐の出力を結合
output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

# 残差接続
from keras import layers

x = ...

# xに変換を適用
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.COnv2D(128, 3, activation='relu', padding='same')(y)
y = layers.COnv2D(128, 3, activation='relu', padding='same')(y)

# 元のxを出力特徴量に追加
y = layers.add([y, x])

x = ...
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.MaxPooling2D(2, strides=2)(y)

# 元のテンソルxをyと同じ形状にするための1 * 1の畳み込みを使った
# 線形ダウンサンプリング
residual = layers.Conv2D(128, , strides=2, padding='same')(x)

# 残差テンソルを出力特徴量に追加
y = layers.add([y, residual])
