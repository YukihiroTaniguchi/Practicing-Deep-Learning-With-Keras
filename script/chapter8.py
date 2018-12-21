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
