### 181203(Mon)  
p.135 - p. 150
chapter5.py : L75-L266
##### *Remember me*  
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
#data batch shape: (20, 150, 150, 3)
#labels batch shape: (20,)

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

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
