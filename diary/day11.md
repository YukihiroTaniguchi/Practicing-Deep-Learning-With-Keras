### 181207(Fri)  
p.175 - p.181
chapter5.py : L576-L688
##### *Remember me*  
CNNのフィルタを可視化する

勾配上昇法  
->空の入力から始めて、  
->CNNの入力画像の値に勾配降下法を適用することで、  
->特定のフィルタの応答を可視化する

```python
# CNNのフィルタを可視化するための損失テンソルの定義
from keras.applications import VGG16
from keras import backend as K

model  = VGG16(weights='imagenet', include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
# 損失関数
loss = K.mean(layer_output[:, :, :, filter_index])
```
```python
#入力に関する損失関数の勾配を取得
#gradientsの呼び出しはテンソル(この場合はサイズ1)のリストを返す
#このため、最初の要素(テンソル)だけを保持する
grads = K.gradients(loss, model.input)[0]
```
```python
#勾配の正規化
#除算の前に1e-5を足すことで、0で除算を回避
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
```
```python
#入力値をNumpy配列で受け取り、出力値をNumpy配列で返す関数
iterate = K.function([model.input], [loss, grads])

import numpy as np
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])
```
```python
#確率的勾配降下法を使って損失値を最大化
#最初はノイズが含まれたグレースケール画像を使用
input_img_data = np.random.random((1, 150, 150, 3)) * 20 +  128.
#勾配上昇法を40ステップ実行
step = 1. #各勾配の更新の大きさ
for i in range(40):
    #損失値と勾配値を計算
    loss_value, grads_value = iterate([input_img_data])
    #損失が最大になる方向に入力画像を調整
    input_img_data += grads_value * 40
```
```python
#出力される画像テンソルを255スケールに変換するユーティリティ関数
def deprocess_image(x):
    #テンソルを正規化: 中心を0、標準偏差を0.1にする
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    #[0, 1]でクリッピング
    #(0より小さい値は0になり、1より大きい値は1になる)
    x += 0.5
    x = np.clip(x, 0, 1)
    #RGB配列に変換
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x
```
```python
#フィルタを可視化するための関数
def generate_pattern(layer_name, filter_index, size=150):

    #ターゲット層のn番目のフィルタの活性化を最大化する損失関数を構築
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    #この損失関数を使って入力画像の勾配を計算
    grads = K.gradients(loss, model.input)[0]

    #正規化トリック: 勾配を正規化
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    #入力値に基づいて損失値と勾配を返す関数
    iterate = K.function([model.input], [loss, grads])

    #最初はノイズが含まれたグレースケール画像を使用
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    #勾配上昇法を40ステップ実行
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)

#層block3_conv1 の0番目のチャネルの応答を最大化するパターン
plt.imshow(generate_pattern('block3_conv1', 0))
```
```python
#4つの畳み込みブロックの最初の層
#(block1_conv1, block2_conv1, block3_conv1, block4_conv1)
layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
for layer_name in layers:
    size = 64
    margin = 5

    #結果を格納する空の画像
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

    for i in range(8):#resultsグリッドの行を順番に処理
        for j in range(8):#resultsグリッドの列を順番に処理

            #layer_nameのフィルタi + (j * 8)のパターンを生成
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            #resultsグリッドの矩形(i, j)に結果を配置
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end,
                    vertical_start: vertical_end, :] = filter_img

    #reslutsグリッドを表示
    plt.figure(figsize=(20, 20))                
    plt.imshow(results)

```
