#### chapter5 まとめ  
##### 畳み込みニューラルネットワーク(CNN)  
- 畳み込み層(Conv2D層)  
-> 特定のパターンを学習したら、他の場所に出現した同様のパターンを認識できる  
-> 小さな局所的パターンからより大きなパターンを学習し、資格概念の抽象化を行うことが可能

幅、高さ、チャネル(深さ)を持つ  
最初の入力層でチャネルはRGBのカラーチャネル(3)だが、パターンの有無を表すもの(より多くのチャネルを持つ-32とか)になる  
-> 特徴マップ(特徴(パターン)がどこに現れているかを示す。チャネルによって示す特徴が異なる)

```python
Conv2D(output_depth, (window_height, window_width))
```
入力特徴マップ  
↓  
3 \* 3 とか 5 \* 5 の入力パッチをずらしていく  
↓  
それぞれの入力パッチを1次元ベクトルに変換する  
↓  
1次元ベクトルをまとめて出力特徴マップを生成

周辺効果  
5 \* 5 の特徴マップに 3 \* 3 の入力パッチを当てて作られるタイルは 3 \* 3 だけ  
-> (幅-1) \* (高さ-1) の大きさになって少し小さくなる  

パディング  
周辺効果によって出力特徴マップの空間次元が損なわれるのを避ける  
->入力特徴マップの上下左右に1行ずつ追加していく

```python
padding="valid"(なし)または"same"(上下左右に1行ずつ追加)
```

ストライド  
ずらす値を変更する(デフォルトは1)

- 最大値プーリング演算  
2 \* 2 ウインドウとストライド2でパッチの最大値をダウンサンプリングする  
->サイズが半分になる

```python
model.add(layers.MaxPooling2D((2, 2)))
```
->最大値プーリング演算を用いてダウンサンプリングをすることで
1. 処理対象の特徴マップの係数を減らすことができる
2. 連続する畳み込みが調べるウィンドウを徐々に大きくすることで、空間フィルタ階層を抽出できる

---
##### 画像分類モデル  
画像データの前処理ステップ
1. 画像ファイルを読み込む
2. JPEGファイルの内容をRGBのピクセルグリッドにデコードする。
3. これらのピクセルを浮動小数点型のテンソルに変換する。
4. ピクセル値(0~255)の尺度を[0, 1]の範囲の値にする。

```python
from keras.preprocessing.image import ImageDataGenerator
#すべての画像を1/255でスケーリング
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,              #ターゲットディレクトリ
    target_size=(150, 150), #すべての画像のサイズを 150 * 150 に変更
    batch_size=20,          #バッチサイズ
    class_mode='binary')    #binary_crossentropy を使用するため、二値のラベルが必要

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
```
学習させる
```python
history = model.fit_generator(train_generator,
                              steps_per_epoch=100, #2000サンプル / 1バッチ20
                              epochs=300,
                              validation_data=validation_generator,
                              validation_steps=50) #1000サンプル / 1バッチ20
```
学習後、モデルを保存
```python
model.save('hoge_model.1')
```

サンプルが少ない場合、データ拡張を行い、サンプル数の水増しをする  
```python
datagen = ImageDataGenerator(rotation_range=40, #画像をランダムに回転させる
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             #画像を水平または垂直にランダムに並行移動させる範囲(幅全体または高さ全体の割合)
                             shear_range=0.2, #等積変形をランダムに適用
                             zoom_range=0.2, #図形の内側をランダムにズーム
                             horizontal_flip=True, #画像の半分を水平方向にランダムに反転
                             fill_mode='nearest') #新たに作成されたピクセルを埋めるための戦略
```

学習済みのCNN

特徴抽出  
-> 1つ前のネットワークが学習した表現に基づいて、新しいサンプルから興味深い特徴量を抽出する  
-> 畳み込みニューラルネットワーク(畳み込みベース)を新しい分類器を訓練する  
-> 全結合層には入力画像のどこにオブジェクトが位置しているかに関する情報はすでに含まれていない  
-> 汎用性の度合いは、その層の深さが重要になる
-> 層が深すぎると抽象的概念(猫の耳、犬の目など)を抽出する
-> 層が浅いと汎用性の高い局所的な特徴マップ(エッジ、色、テクスチャなど)を抽出する
-> モデルの最初のほうにあるいくつかの層だけを使用したほうが良い

VGG16のインスタンス化
```python
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', #重みのチェックポイントを指定
                  include_top=False, #ネットワークの出力側にある全結合分類機を含めるかどうか
                  input_shape=(150, 150, 3)) #ネットワークに供給する画像テンソルの形状
```

1. データ拡張を行わない高速な特徴抽出  
学習済みの畳み込みベースを使って特徴量を抽出

```python
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
pwd
base_dir = '/Users/hoge/Documents/Practicing-Deep-Learning-With-Keras/data/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory,
                                            target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break

    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)
```

-> 学習済みのデータから抽出した学習済みの特徴量を全結合層に投入
```python
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))
```  

2. データ拡張を行う特徴抽出  
学習済みの畳込みベースに全結合層を追加し、拡張したデータを投入

```python
from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False
#畳み込みベースを凍結

print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)
#検証データは水増ししない

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50,
                              verbose=2)

```

##### ファインチューニング
-> 凍結された畳み込みベースの出力側の層をいくつか解凍し、モデルの新しく追加された部分と回答した層の両方で訓練を行う

ファインチューニングの手順
1. 訓練済みのベースネットワークの最後にカスタムネットワークを追加する
2. ベースネットワークを凍結する
3. 追加した部分の訓練を行う
4. ベースネットワークの一部の層を解凍する
5. 解凍した層と追加した部分の訓練を同時に行う

出力側の層のみをファインチューニングする
-> 理由は2点

* 入力側は汎用的で再利用可能な特徴量をエンコードしている  
-> 出力側はより具体的な特徴量をエンコードしている  
-> 入力側の層をファインチューニングすれば、収穫逓減が早まる
* 訓練対称のパラメータの数が増えるほど、過学習のリスクが高まる
-> なるべく少なくする
例 : block5_conv1以降の層のみ凍結を解除する

```python
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
```
##### CNNの可視化
1. 中間層の出力を可視化する
層の出力  
-> 層の活性化と呼ばれる

画像の前処理  
入力する画像を4次元テンソルに変換する
```python
from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
#3次元テンソルを4次元テンソルに変換

img_tensor /= 255.
print(img_tensor.shape)
```
sequential モデル  
-> 入力数: 1, 出力数: 1

keras モデル  
-> 入力数: 制限なし, 出力数: 制限なし

一つの画像を入力し、複数の層における出力をはきだす
```python
from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
#学習済みのモデルの0-8の層の出力

activation_model = models.model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
activations[0].shape[1]#148
first_layer_activation = activations[0]
first_layer_activation.shape#(1, 148, 148, 32)
activations[1].shape#(1, 74, 74, 32)
activations[2].shape#(1, 72, 72, 64)
activations[3].shape#(1, 36, 36, 64)
activations[4].shape#(1, 34, 34, 128)
activations[5].shape#(1, 17, 17, 128)
activations[6].shape#(1, 15, 15, 128)
activations[7].shape#(1, 7, 7, 128)
```

中間層の活性化ごとにすべてのチャネルを可視化
```python
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in  zip(layer_names, activations):
    #特徴マップに含まれる特徴量の数(チャネル数)
    n_features = layer_activation.shape[-1]

    #特徴マップの形状(1, size, size, n_features)
    size = layer_activation.shape[1]

    #この行列で活性化のチャネルをタイル表示
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    #各フィルタを1つの大きな水平グリッドでタイル表示
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :,
                                             col * images_per_row + row]

            # 特徴量の見た目を良くするための後処理
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(false)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
```

最初の層は情報のほぼすべてが活性化に含まれている  
層が進むに従い活性化は徐々に抽象化されていき、高レベルな概念を抽象化するようになる  
画像クラスに関連する情報が増えていく  
空のフィルタが増えていく  
-> 活性化の疎性  
そのフィルタにエンコードされているパターンが入力画像から検出されないことを意味する  
入力値に関する情報は先へ進むほど減っていき目的地に関する情報が増えていく  
-> 情報蒸留パイプライン  
無関係な情報が取り除かれ、有益な情報が誇張され、整えられていく。

2. CNNのフィルタを可視化する

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
3. クラス活性化をヒートマップとして可視化  

Grad-CAM  
「入力画像によって様々なチャネルがどれくらい強く活性化されるか」を表す空間マップ  
\*  
「そのクラスにとって各チャネルがどれくらい重要か」を表す値で重み付けする  
->  
「入力画像によってそのクラスがどれくらい強く活性化されるか」を表す空間マップが得られる  

VGG16モデルに合わせて入力画像を前処理
```python
model = VGG16(weights='imagenet')
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

img_path = '/Users/yukihiro/Documents/Practicing-Deep-Learning-With-Keras/others/creative_commons_elephant.jpg'

#ターゲット画像を読み込む : imgはサイズが224 × 224のPIL画像
img = image.load_img(img_path, target_size=(224, 224))

#xは形状が(224, 224, 3)のfloat32型のNumPy配列
x = image.img_to_array(img)

#(1, 224, 224, 3)のバッチに変換するために次元を追加
x = np.expand_dims(x, axis=0)

#バッチの前処理(チャネルごとに色を正規化)
x = preprocess_input(x)
```

Grad-CAM
```python
preds = model.predict(x)
afrtican_elepaht_output = model.output[:, 386]

#VGG16の最後の畳み込み層であるblock5_conv3の出力特徴マップ
last_conv_layer = model.get_layer('block5_conv3')

#block5_conv3の出力特徴マップでの「アフリカゾウ」クラスの勾配
grads = K.gradients(afrtican_elepaht_output, last_conv_layer.output)[0]

#形状が(512, )のベクトル
#各エントリは特定の特徴マップチャネルの勾配の平均強度
pooled_grads = K.mean(grads, axis=(0, 1, 2))

#2頭のアフリカゾウのサンプル画像に基づいて、pooled_gradsと
#block5_conv3の出力特徴マップの値にアクセスするための関数
iterate = K.function([model.input],
                     [pooled_grads, last_conv_layer.output[0]])

#これら2つの値をNumPy配列として取得
pooled_grads_value, conv_layer_output_value = iterate([x])

#「アフリカゾウ」クラスに関する「このチャネルの重要度」を
#特徴マップ配列の各チャネルに掛ける
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

#最終的な特徴マップのチャネルごとの平均値が
#クラスの活性化のヒートマップ
heatmap = np.mean(conv_layer_output_value, axis=-1)

#ヒートマップの後処理
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
import matplotlib.pyplot as plt
plt.matshow(heatmap)
```

元の画像にスーパーインポーズ
```python
import cv2

#cv2を使って元の画像を読み込む
img = cv2.imread(img_path)

#もとの画像と同じサイズになるようにヒートマップのサイズを変更
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

#ヒートマップをRGBに変換
heatmap = np.uint8(255 * heatmap)

#ヒートマップをもとの画像に適用
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#0.4はヒートマップの強度係数
superimposed_img = heatmap * 0.4 + img

#保存
cv2.imwrite('/Users/yukihiro/Documents/Practicing-Deep-Learning-With-Keras/others/elephant_cam.jpg', superimposed_img)
```
