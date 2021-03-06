### 181221(Fri)    
p.295 - p.301
chapter8.py : L1 - L121
##### *Remember me*  
DeepDream を Keras で実装する  
```python
# 学習済みのInception V3 モデルを読み込む
from keras.applications import inception_v3
from keras import backend as K

# ここではモデルを訓練しないため、訓練関連の演算はすべて無効にする
K.set_learning_phase(0)

# InceptionV3ネットワークを畳み込みベース無しで構築する
# このモデルは学習済みのImageNetの重み付きで読み込まれる
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
```

DeepDream の構成
```python
# 層の名前を係数にマッピングするディクショナリ。この係数は最大化の対象と
# なる損失値にその層の活性化がどれくらい貢献するのかを表す。これらの層の
# 名前は組み込みのInception V3アプリケーションにハードコーディングされて
# いることに注意。すべての層の名前はmodel.summary()を使って確認できる
layer_contributions = {
    'mixed2' : 0.2,
    'mixed3' : 3.,
    'mixed4' : 2.,
    'mixed5' : 1.5,
}
```

損失値を保持するテンソルの定義  
最大化の対象となる損失値を定義
```python
# 最大化の対象となる損失値を定義
# 層の名前を層のインスタンスにマッピングするディクションナリを作成
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# 損失値を定義
loss = K.variable(0.)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]

    # 層の出力を取得
    activation = layer_dict[layer_name].output

    scaling = K.prod(K.cast(K.shape(activation), 'float32'))

    # 層の特徴量のL2ノルムをlossに加算
    # 非境界ピクセルのみをlossに適用することで、周辺効果を回避
    loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling
```

勾配上昇法のプロセスを設定
```python
# 生成された画像(ドリーム)を保持するテンソル
dream = model.input

# ドリームの損失関数の勾配を計算
grads = K.gradients(loss, dream)[0]

# 勾配を正規化(重要)
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

# 入力画像に基づいて損失と勾配の値を取得するKeras関数を設定
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grads_value = outs[1]
    return loss_value, grads_value

# 勾配上昇法を指定された回数に渡って実行する関数
def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x
```

DeepDreamのアルゴリズムを定義  
オクターブ = 尺度  
尺度はそれぞれ1つ前の尺度よりも1.4倍(40%)ずつ大きくなる  
-> 最初は小さな画像を処理することから始める

異なる尺度にわたって勾配上昇法を実行
```python
import numpy as np
# これらのハイパーパラメータでイリリな値を試してみることでも、
# 新しい効果が得られる

step = 0.01         # 勾配上昇法のステップサイズ
num_octave = 3      # 勾配上昇法を実行する尺度の数
octave_scale = 1.4  # 尺度間の拡大率
iterations = 20     # 尺度ごとの上昇ステップの数

# 損失値が10を超えた場合は見た目がひどくなるのを避けるために購買上昇法を中止
max_loss = 10.

# 使用したい画像へのパスに置き換える
base_image_path = ''

# ベースとなる画像をNumPy配列に読み込む
img = preprocess_image(base_image_path)

# 勾配上昇法を実行する様々な尺度を定義する形状タプルのリストを準備
original_shape  = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i))
                   for dim in original_shape])
    successive_shapes.append(shape)

# 形状リストを逆にして昇順になるようにする
successive_shapes = successive_shapes[::-1]

# 画像のNumPy配列のサイズを最も小さな尺度に変換
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    # ドリーム画像を拡大
    img = resize_img(img, shape)
    # 勾配上昇法を実行してドリーム画像を加工
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    # 元の画像を縮小したものを拡大 : 画像が画素化される
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    # このサイズでの元の画像の高品質バージョンを計算
    same_size_original = resize_img(original_img, shape)
    # これらの2つの差分が、拡大時に失われるディテールの量
    lost_detail = same_size_original - upscaled_shrunk_original_img
    #  失われたディテールをドリーム画像に再注入
    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
save_img(img, fname='final_dream.png')
```

補助関数一覧

```python
import scipy
from keras.preprocessing import image

# 画像のサイズを変更
def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)

# 画像を保存
def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)

# 画像を開いてサイズを変更し、Inception V3が処理できるテンソルに変換
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

# テンソルを有効な画像に変換
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x
```
