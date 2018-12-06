### 181206(Thu)  
p.167 - p.174
chapter5.py : L490-L574
##### *Remember me*  
CNNの中間出力の可視化

層の出力  
-> 層の活性化と呼ばれる

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
Sequential モデル  
-> 入力数: 1, 出力数: 1

Keras モデル  
-> 入力数: 制限なし, 出力数: 制限なし

一つの画像を入力し、複数の層における出力をはきだす
```python
from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
#学習済みのモデルの0-8の層の出力

activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
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
    plt.grid(False)
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
