import numpy as np

# 8.1.3 サンプリング戦略の重要性

# 異なる温度での確率分布の再荷重
# original_distribution は確率値からなる1次元のNumPy配列
# 確率値の総和は1でなければならない
# temperature は出力分布のエントロピーを定量化する係数
def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)

    # 再荷重された元の確率分布を返す
    # 新しい確率分布の総和は1にならない可能性があるため、
    # 新しい分布を取得するために総和で割っている
    return distribution / np.sum(distribution)


# 8.1.4 LSTMによる文字レベルのテキスト生成
import keras
import numpy as np

path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length:', len(text))

# 文字のシーケンスのベクトル化
maxlen = 60     # 60文字のシーケンス
step = 3        # 3文字お気に新しいシーケンスをサンプリング
sentences = []  # 抽出されたシーケンスを保持
next_chars = [] # 目的地(次に来る文字)を保持

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print('Number of sequences:', len(sentences))

# コーパスの一意な文字のリスト
chars = sorted(list(set(text)))
print('Unique characters:', len(chars))

print(chars)

# これらの文字をリストcharsのインデックスにマッピングする
char_indices = dict((char, chars.index(char)) for char in chars)

print(char_indices)

print('Vectorization...')

# one-hotエンコーディングを適用して文字を二値の配列に格納
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# 次の文字を予測する単相LSTMモデル
from keras import layers

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

# モデルのコンパイル設定
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy')

# モデルの予測に基づいて次の文字をサンプリングする関数
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# テキスト生成ループ
import random
import sys

# モデルを60エポックで訓練
for epoch in range(1, 60):
    print('epoch', epoch)

    # 1エポックでデータを学習
    model.fit(x, y, batch_size=128, epochs=1)

    # テキストシードをランダムに選択
    start_index = random.randint(0, len(text) - maxlen -1)
    gererated_text = text[start_index: start_index + maxlen]
    print('--- Gererating with sedd: "' + gererated_text + '"')

    # ある範囲内の異なるサンプリング温度を試してみる
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(gererated_text)

        # 400文字を生成
        for i in range(400):

            # これまでに生成された文字にone-hotエンコーディングを適用
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(gererated_text):
                sampled[0, t, char_indices[char]] = 1.

            # 次の文字をサンプリング
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            gererated_text += next_char
            gererated_text = gererated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()

# 8.2.1 DeppDreamをKerasで実装する
# 学習済みのInception V3 モデルを読み込む
from keras.applications import inception_v3
from keras import backend as K

# ここではモデルを訓練しないため、訓練関連の演算はすべて無効にする
K.set_learning_phase(0)

# InceptionV3ネットワークを畳み込みベース無しで構築する
# このモデルは学習済みのImageNetの重み付きで読み込まれる
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

# DeepDream の構成
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

model.summary()

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

# 勾配上昇法のプロセス
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

# 異なる尺度にわたって勾配上昇法を実行
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
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    # このサイズでの元の画像の高品質バージョンを計算
    same_size_original = resize_img(original_img, shape)
    # これらの2つの差分が、拡大時に失われるディテールの量
    lost_detail = same_size_original - upscaled_shrunk_original_img
    #  失われたディテールをドリーム画像に再注入
    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
save_img(img, fname='final_dream.png')

# 補助関数
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

# 8.3.3 Keras でのニューラルスタイル変換
# 画像パス等の定義
from keras.preprocessing.image import load_img, img_to_array

# 変換したい画像へのパス
target_image_path = 'data/img/portrait.jpg'

# スタイル画像へのパス
style_reference_image_path = 'data/img/transfer_style_reference.jpg'

# 生成する画像のサイズ
width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)


# 画像の読み込み、前処理、後処理を行う補助関数
import numpy as np
from keras.applications import vgg19

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    # ImageNetから平均ピクセル値を取り除くことにより、中心を0に設定
    # これにより、vgg19.preprocess_inputによって実行される変換が逆になる
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # 画像を'BGR'から'RGB'に変換
    # これもvgg19.preprocess_inputの変換を逆にするための措置
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

# 学習済みのVGG19ネットワークを読み込み、3つの画像に適用
from keras import backend as K

target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))

# 生成された画像を保持するプレースホルダ
combination_image = K.placeholder((1, img_height, img_width, 3))

# 3つの画像を1つのバッチにまとめる
input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)

# 3つの画像からなるバッチを入力として使用するVGG19モデルを構築
# このモデルには、学習済みのImageNetの重みが読み込まれる
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)
print('Model loaded.')

# コンテンツの損失関数
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# スタイルの損失関数
# グラム行列を求める補助関数
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# 過度の画素化を防ぐ正則化のための全変動損失関数
def total_variation_loss(x):
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] -
                 x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] -
                 x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# 最小化の対象となる最終的な損失関数を定義
# 層の名前を活性化テンソルにマッピングするディクショナリ
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

content_layer = 'block5_conv2' # コンテンツの損失関数に使用する層の名前

style_layers = ['block1_conv1', # スタイルの損失関数に使用する僧の名前
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
# 損失関数の加重平均の重み
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

# すべてのコンポーネントをこのスカラー変数に追加することで、損失関数を定義
loss = K.variable(0.)

# コンテンツの損失関数を追加
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features, combination_features)

# 各ターゲット層のスタイルの損失関数を追加
for layer_name in style_layers:
    layer_features =outputs_dict[layer_name]
    style_reference_freatures = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_freatures, combination_features)
    loss += (style_weight / len(style_layers)) * sl

# 全変動損失関数を追加
loss += total_variation_weight * total_variation_loss(combination_image)

# 勾配降下法のプロセスを定義
# 損失関数をもとに、生成された画像の勾配を取得
grads = K.gradients(loss, combination_image)[0]

#現在の損失関数の値と勾配の値を取得する関数
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

# このクラスは、損失関数の値と勾配の値を2つのメソッド呼び出しを通じて取得で
# きるようにfetch_loss_and_gradsをラッピングする。この2つのメソッド呼び出し
# は、ここで使用するSciPyのオプティマイザによって要求される
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.garads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# スタイル変換ループ
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

result_prefix = 'style_trasfer_result'
iterations = 20

# 初期状態 : ターゲット画像
x = preprocess_image(target_image_path)

# 画像を平坦化 : scipy.optimize.fmin_l_bfgs_bは1次元ベクトルしか処理できない
x = x.flatten()

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()

    # ニューラルスタイル変換の損失関数を最小化するために
    # 生成された画像のピクセルに渡ってL-BFGS最適化を実行
    # 損失関数を計算する関数と勾配を計算する関数を2つの別々の引数として
    # 渡さなければならないことに注意
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # この時点の生成されたが画像を保存
    img = x.copy().reshape((img__height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))


# 8.4.3 変分オートエンコーダ(VAE)
import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np

img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 2 #潜在空間の次元数 : 2次元平面

input_img = keras.Input(shape=img_shape)

x = layers.Conv2D(32, 3,
                  padding='same', activation='relu')(input_img)

x = layers.Conv2D(64, 3,
                  padding='same', activation='relu',
                  strides=(2, 2))(x)

x = layers.Conv2D(64, 3,
                  padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

# 入力画像はこれら2つのパラメータにエンコードされる
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# 潜在空間の点を画像にマッピングするVAEデコーダネットワーク
# この入力でzを供給
decoder_input = layers.Input(K.int_shape(z))

# 入力を正しい数のユニットにアップサンプリング
x = layers.Dense(np.prod(shape_before_flattening[1:]),
                 activation='relu')(decoder_input)

# 最後のFlatten層の直前の特徴マップと同じ形状
x = layers.Reshape(shape_before_flattening[1:])(x)

# Conv2DTranspose層とConv2D層を使って
# 元の入力画像と同じサイズの特徴マップに変換
x = layers.Conv2DTranspose(32, 3,
                           padding='same', activation='relu',
                           strides=(2, 2))(x)
x =layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)

# decoder_input をデコードされた画像に変換するデコーダモデルをインスタンス化
decoder = Model(decoder_input, x)

# このモデルをzに適用してデコードされたzを復元
z_decoded = decoder(z)

class CustomVariationalLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    # カスタム層の実装ではcallメソッドを定義する
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        # この出力は使用しないが、層は何かを返さなければならない
        return x

y = CustomVariationalLayer()([input_img, z_decoded])

# VAEの訓練
from keras.datasets import mnist

vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape + (1,))

vae.fit(x=x_train, y=None,
        shuffle=True,
        epochs=10,
        batch_size=batch_size,
        validation_data=(x_test, None))


# decoderネットワークを使って任意の潜在空間のベクトルを画像に変換
# 2次元の潜在空間から点のグリッドを抽出し、画像にデコード
import matplotlib.pyplot as plot
from scipy.stats import norm

# 15 * 15 の数字のグリッドを表示(数字は合計で255個)
n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# SciPyのppf関数を使って線形空間座標を変換し、潜在変数zの値を生成
# (潜在空間の前はガウス分布であるため)
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.llisapce(0.05, 0.95, n))

for i, yi enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        # 完全なバッチを形成するためにzを複数回繰り返す
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        z_sample = np.expand_dims(z_sample, axis=0)
        # バッチを数字の画像にデコード
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        # バッチの最初の数字を28 * 28 * 1 から28 * 28 に変形
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Grays_r')
plt.show()

import keras
from keras import layers
import numpy as np
latent_dim = 32
height = 32
width = 32
channels = 3

generator_input = keras.Input(shape=(latent_dim,))

# 入力を16 * 16、128チャネルの特徴マップに変換
x = layers.Dense(128 * 16 * 16)(generator_input)
y = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

# 畳み込み層を追加
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# 32 * 32 にアップサンプリング
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

# さらに畳み込み層を追加
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# 32 * 32 1チャネル(CIFAR10の画像の形状)の特徴マップを生成
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)

# generatorモデルをインスタンス化 :
# 形状が(latent_dim,)の入力を形状が(32, 32, 3)の画像にマッピング
generator = keras.models.Model(generator_input, x)
generator.summary()

# GANの判別者ネットワーク
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

# ドロップアウト層を1つ追加 : 重要なトリック!
x = layers.Dropout(0.4)(x)

# 分類層
x = layers.Dense(1, activation='sigmoid')(x)

# discriminatorモデルをインスタンス化
形状が(32, 32, 3)の入力で二値分類(fake/reak)を実行
discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

# オプティマイザで勾配刈り込みを使用し(clipvalue)、
# 訓練を安定させるために学習率減衰を使用(decay)
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008,
                                                   clipvalue=1.0,
                                                   decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer,
                      loss='binary_crossentropy')

# discriminatorの重みを訓練不可能に設定(これはganモデルにのみ適用される)
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0,
                                          decay=1e-8)

gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
