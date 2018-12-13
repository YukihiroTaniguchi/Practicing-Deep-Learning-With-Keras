### 181213(Thu)    
p.217 - p.226
chapter6.py : L514 - L667
##### *Remember me*  
時系列サンプルとそれらのターゲットを生成するデータジェネレーター  

data : もとの浮動小数点数型のデータからなる配列  
lookback : 入力データのtimestepsをいくつさかのぼるか  
delay : Targetのtimestepsをいくつ進める化  
min_index, max_index : 抽出するtimestepsの上限と下限を表すdata配列のインデックス。データの一部を検証とテストのためにとっておくのに役立つ  
shuffle : サンプルをシャッフルするのか、それとも時間の順序で抽出するのか  
batch_size : バッチひとつあたりのサンプル数  
step : データをサンプリングする時の期間(単位はtimesteps)データポイントを1時間ごとに1つ抽出するために6に設定
```python
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index,
                                     size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
      yield samples, targets
```
訓練、検証、テストジェネレータの準備

```python
lookback = 1440
step = 6
delay = 144
batch_size = 128

#訓練ジェネレータ
train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

#検証ジェネレータ
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)

#テストジェネレータ
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

#検証データセット全体を調べるためにval_genから抽出するtimestepsの数
val_steps = (300000 - 200001 - lookback) // batch_size

#テストデータセット全体を調べるためにtest_genから抽出するtimestepsの数
test_steps = (len(float_data) - 300001 - lookback) // batch_size
```

常識的なベースラインを設定する  
Aクラスのインスタンス : 90%  
Bクラスのインスタンス : 10%  
-> 常識的なアプローチ 常にAクラスを予測する  
-> 正解率90%
これを超えるようにする

RNNなどの複雑で計算負荷の高い機械学習モデルを調べる再には、単純で計算不可の低いモデルを試してみる
-> 計算負荷の低い全結合モデル
```python
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step,
                                      float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
```

GRU層  
-> LSTM層よりも実行コストが掛からない  
-> LSTM層ほど表現力がない

```python
#GRUベースのモデルの訓練と評価
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))#特徴量の数
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
```

リカレントドロップアウト
-> timestepsごとにドロップアウトマスクをランダムに変化させるのではなく  
-> すべてのtimestepsごとに同じドロップアウトマスクを適用する  
-> すべてのtimestepsで同じドロップアウトマスクを使用すると、ネットワークが時間の流れに沿って学習誤差を正しく伝播できる
