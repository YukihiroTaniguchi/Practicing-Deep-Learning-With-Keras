### 181218(Tue)    
p.259 - p.262
chapter7.py : L245 - L292
##### *Remember me*  
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
