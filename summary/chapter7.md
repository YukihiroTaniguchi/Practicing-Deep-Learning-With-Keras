#### chapter7 まとめ  
##### Keras Functional API
Sequentialモデルだけではカバーできないものがある
- 他入力モデル  
-> マルチモーダル入力    
- 他出力モデル  
- グラフ形式のモデル   
-> 有効批准回グラフとして構造化されたネットワーク  
-> Inception モジュール
-> 残差接続

Functional API でできる


Sequentialモデルを Functional API で実装
```python
from keras import Input, layers
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
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 64)                0         
_________________________________________________________________
dense_5 (Dense)              (None, 32)                2080      
_________________________________________________________________
dense_6 (Dense)              (None, 32)                1056      
_________________________________________________________________
dense_7 (Dense)              (None, 10)                330       
=================================================================
Total params: 3,466
Trainable params: 3,466
Non-trainable params: 0
_________________________________________________________________
```

Keras は input_tensor から output_tensor までの間にあるそうをすべて取得し、  
それをグラフにまとめる  
-> 中間層の情報が不要なのは output_tensor が input_tensor を  
-> 繰り返し変換することによってい取得されたものだから

多入力モデル
```python
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
model.summary()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

# 多入力モデルへのデータの供給
import numpy as np
num_samples = 1000
max_length = 100

# ダミーのNumPyデータを生成
text = np.random.randint(1, text_vocabulary_size,
                         size=(num_samples, max_length))

question = np.random.randint(1, question_vocabulary_size,
                             size=(num_samples, max_length))

# 答えに(整数ではなく)one-hotエンコーディングを適用(target)
answers = np.zeros(shape=(num_samples, answer_vocabulary_size))
indices = np.random.randint(0, answer_vocabulary_size, size=num_samples)
for i, x in enumerate(answers):
    x[indices[i]] = 1

# 入力リストを使った適合
model.fit([text, question], answers, epochs=10, batch_size=128)

# 入力ディクショナリを使った適合(入力に名前をつける場合)
model.fit({'text': text, 'question': question}, answers,
         epochs=10, batch_size=128)
```
多出力モデル  
損失値を一つにまとめる  
-> 損失値の総和を求める  
-> 損失値の尺度が異なるのでそれぞれの損失値に重みを割り当てる
```python
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

```
有効非巡回グラフ

Inceptionモジュール  
1 * 1 の畳込み  
-> pw畳み込み(pointwise convolution)
```python
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
```
残差接続  
手前にある層の出力を後ろにある層の入力にすることで  
逐次的なネットワークにショートカットを作成する  
活性化のサイズが同じであることを前提に  
後ろにある層の活性化と合計する

```python
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
```

層の共有  
-> 複数の分岐を共有するモデルを構築できる  
-> Siamese LSTM or 共有LSTM と呼ばれている

```python
from keras import layers
from keras import Input
from keras.models import Model

# 単一のLSTM層を一度だけインスタンス化
lstm = layers.LSTM(32)

# モデルの浸り側の分岐を構築:
# 入力はサイズが128のベクトルからなる可変長のシーケンス
left_input = Input(shape=(None, 128))
left_output = lstm(left_input)

# モデルの右側の分岐を構築:
# 既存の層のインスタンスを呼び出すと、その重みを再利用することになる
right_input = Input(shape=(None, 128))
left_output = lstm(right_input)

# 最後に分類器を構築:
# 既存の層のインスタンスを呼び出すと、その重みを再利用することになる
merged = layers.concatenate([left_output, right_output], axis = -1)
predictions = layers.Dense(1, activation='sigmoid')(merged)

# モデルのインスタンス化と訓練:
# このようなモデルを訓練するときには、
# LSTM層の重みガ両方の入力に基づいて更新される
model = Model([left_input, right_input], predictions)
model.fit([left_data, right_data], targets)
```
層としてモデルを使用する
```python
from keras import layers
from keras import applications
from keras import Input

# ベースとなる画像処理モデルはXceptionネットワーク(畳み込みベースのみ)
xception_base = applications.Xception(weights=None, include_top=False)

# 入力は250*250のRGB画像
left_input = Input(shape=(250, 250, 3))
right_input= Input(shape=(250, 250, 3))

# 同じビジョンモデルを2回呼び出す
left_features = xception_base(left_input)
right_features = xception_base(right_input)

# マージ後の特徴量には、右の視覚フィードと左の視覚フィードの情報が含まれている
merged_features = layers.concatenate([left_features, right_features], axis=-1)
```

##### Keras のコールバック
コールバックを使ってモデルを制御する

- モデルのチェックポイント化  
-> 訓練中の様々な時点ででモデルの現在の重みを保存する
- 訓練の中止  
-> 検証データでの損失値がそれ以上改善しなくなったところで、訓練を中止する
- 特定のパラメータの動的な調整  
-> 訓練中のパラメータの動的な調整  
- 訓練と検証の指標を記録  
-> 訓練と検証の指標をログに記録するか、可視化する


ModelCheckpoint コールバックと EarlyStopping コールバック
```python
import keras

# コールバックはfitのcallbacksパラメータを通じてモデルに渡される
# コールバックはいくつ指定してもよい
callbacks_list = [
    # 改善が止まったら訓練を中止
    keras.callbacks.EarlyStopping(
        monitor='val_acc', # 検証データでのモデルの正解率を監視
        patience=1,        # 2エポック以上にわたって正解率が
    ),                      # 改善しなければ訓練を中止
    # エポックごとに現在の重みを保存
    keras.callbacks.ModelCheckpoint(
        filepath='others/my_model.h5', # モデルの保存先となるファイルパス
        monitor='val_loss',     # これらの2つの引数は、val_lossが改善した場合
        save_best_only=True,    # を覗いてモデルファイルを上書きしないこと
    )                           # を意味する ; 改善した場合は、訓練全体
]                               # で最も良いモデルを保持する

# この場合は正解率を監視するため、
# 正解率はモデルの指標の一部でなければならない
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

# このコールバックはval_lossとval_accを監視するため、
# fit呼び出しにvalidation_dataを指定する必要がある
model.fit(x y,
         epochs=10,
         batch_size=32,
         callbacks=callbacks_list,
         validation_data=(x_val, y_val))
```

ReduceLROnPlateau コールバック
```python
callbacks_list = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', # モデルの検証データセットでの損失値を監視
        factor=0.1,         # コールバック餓鬼道したら学習率を10で割る
        patience=10,        # 検証データでの損失値が10エポックにわたって
    )                       # 改善しなかった場合はコールバックを起動
]

# このコールバックはval_lossを監視するため、
# fit呼び出しにvalidation_dataを指定する必要がある
model.fit(x, y,
          epochs=10,
          batch_size=32,
          callbacks=callbacks_list,
          validation_data=(x_val, y_val))
```

カスタムコールバックの作成  
作成可能メソッド
```python
on_epoch_begin #各エポックの最初に呼び出される
on_epoch_end   #各エポックの最後に呼び出される

on_batch_begin #各バッチを処理する直前に呼び出される
en_batch_end   #各バッチを処理した直後に呼び出される

on_train_begin #訓練の最初に呼び出される
on_train_end   #訓練の最後に呼び出される
```

単純なカスタムコールバック
```python
import keras
import numpy as np

class activationlogger(keras.callbacks.callbacks):
    def set_model(self, model):

        # 訓練の前に親モデルによって呼び出され
        # 呼び出し元のモデルをコールバックに通知
        self.model = model

        layer_outputs = [layer.output for layer in model.layers]

        # 各層の活性化を返すモデルインスタンス
        self.activations_model = keras.models.model(model.input,
                                                    layer_outputs)

    def on_epoch_end(self, epoch, logs=none):
        if self.validation_data is none:
            raise runtimeerror('requires validation_data.')

            # 検証データの最初の入力サンプルを取得
            validation_sample = self.validation_data[0][0:1]
            activations = self.activations_model.predict(validation_sample)

            # 配列をディスクに保存
            f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
            np.savez(f, activations)
            f.close()
```

##### TensorBoard の操作
アイデア  
->  
実験  
->  
結果  
のループが開発

結果からアイデアを出すには？  
TensorBoard
- 訓練中に指標を視覚的に監視
- モデルのアーキテクチャの可視化
- 活性化と勾配のヒストグラムの可視化
- 埋め込みと勾配のヒストグラムの可視化
- 埋め込みを3次元で調査

TensorBoardコールバックを使ってモデルを訓練
```python
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='others/my_log_dir',     # ログファイルはこの場所に書き込まれる
        histogram_freq=1,                # 1エポックごとに活性化ヒストグラムを記録
        embeddings_freq=1,               # 1エポックごとに埋め込みデータを記録
        embeddings_layer_names=['embed'],
    )
]

history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=callbacks)
```
シェルで  
```
$tensorboard --logdir=others/my_log_dir
```

HISTOGRAMSの表示

PROJECTORタブ  
埋め込み層を選択した次元のみで可視化できる

GRAPHSタブ




##### 最先端のモデルを開発するための重要なベストプラクティス

正規化  
-> バッチ正規化

BatchNormalization層
axisパラメータはデフォルトで -1  
data_format が channels_last の場合はそのままで  
data_format が channels_first の場合は特徴軸は 1 であるため 1 を設定

```python
# バッチ正規化
normalized_data = (data - np.mean(data, axis=...)) / np.std(data, axis=...)
# 上をモジュール

# 畳み込み層の後
conv_model.add(layers.Conv2D(32, 3, activation='relu'))
conv_model.add(layers.BatchNormalization())

# 全結合層
dense_model.add(layers.Dense(32, activation='relu'))
dense_model.add(layers.BatchNormalization())
```

dw畳み込み層  
入力の空間的な位置どうしは高い相関関係にあるものの、異なるチャネルどうしはほぼ独立している と想定される場合  
-> 入力の各チャネルで空間畳み込み演算を別々に実行した後、  
-> pw畳み込み(1 * 1畳み込み)演算を通じて出力チャネルを連結する
-> 限られた量のデータで小さなモデルを一から訓練する場合に有効

```python
from keras.models import Sequential, Model
from keras import layers

height = 64
width = 64
channels = 3
num_classes = 10

model = Sequential()
model.add(layers.SeparableConv2D(32, 3, activation='relu',
                                 input_shape=(height, width, channels,)))
model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.MaxPooling2D(2))

model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.MaxPooling2D(2))

model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy')

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
separable_conv2d_1 (Separabl (None, 62, 62, 32)        155       
_________________________________________________________________
separable_conv2d_2 (Separabl (None, 60, 60, 64)        2400      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 30, 30, 64)        0         
_________________________________________________________________
separable_conv2d_3 (Separabl (None, 28, 28, 64)        4736      
_________________________________________________________________
separable_conv2d_4 (Separabl (None, 26, 26, 128)       8896      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 13, 13, 128)       0         
_________________________________________________________________
separable_conv2d_5 (Separabl (None, 11, 11, 64)        9408      
_________________________________________________________________
separable_conv2d_6 (Separabl (None, 9, 9, 128)         8896      
_________________________________________________________________
global_average_pooling2d_1 ( (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 32)                4128      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                330       
=================================================================
Total params: 38,949
Trainable params: 38,949
Non-trainable params: 0
_________________________________________________________________
```

ハイパーパラメータの最適化
1. ハイパーパラメータの集まりを(自動的に)選択する
2. 対応するモデルを構築する
3. モデルに訓練データを学習させ、検証データで最終的な性能を測定する
4. 次に試すハイパーパラメータの集まりを(自動的に)選択する
5. 手順2~3を繰り返す
6. 最後に、テストデータでモデルの性能を測定する

ハイパーパラメータの更新はかなり難題
- フィードバックを計算するには、新しいモデルを作成し、データセットでーから訓練する必要があります
- ハイパーパラメータ空間は一般に離散的な决定で構成されており、連続的でも微分可能でもない

Hyperopt
-> Hyperas

※検証データを使って計算されたフィードバックに基づいて更新されるため  
-> 実質的にハイパーパラメータを検証データで訓練することになる  
-> 検証データの過学習に気をつける

モデルのアンサンブル
より良い予測値を生成するために様々なモデルの予測値をプーリングする

```python
# 4種類のモデルを使って最初の予測値を計算
pred_a = model_a.predict(x_val)
pred_b = model_b.predict(x_val)
pred_c = model_c.predict(x_val)
pred_d = model_d.predict(x_val)

# この新しい予想配列は最初の予測値よりも正確なはず
final_pred = 0.25 * (pred_a + pred_b + pred_c + pred_d)

# これらの重み(0.5、0.25、0.1、0.15)は実験的に学習されたもの
final_pred = 0.5 * pred_a + 0.25 * pred_b + 0.1 * pred_c + 0.15 * pred_d
```
できるだけ良いモデル  
できるだけ異なるモデル  
を使用する

例 : ランダムフォレストや勾配ブースティング木とかの決定木 + ディープラーニングネットワークのアンサンブル

ディープラーニングとシャローラーニングを融合した Wide and Deep カテゴリのモデル
