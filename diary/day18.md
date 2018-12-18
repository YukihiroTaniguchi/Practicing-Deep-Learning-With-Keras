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
