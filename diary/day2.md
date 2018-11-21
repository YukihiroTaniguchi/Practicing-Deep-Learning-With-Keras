### 181121(Wen)  
p.28 - p.31  
chapter2.py : L1-L211
##### *Remember me*  
MNIST(Mixed National Institute of Standards and Technology)

分類問題のカテゴリ  
-> クラス  
データポイント  
-> サンプル、標本
特定のサンプルに関連付けられているクラス
-> ラベル

python keras tensorflowのバージョン確認
```
import sys
print(sys.version)
import keras
print(keras.__version__)
import tensorflow
print(tensorflow.__version__)
```

訓練データセット  
テストデータセット

層 (layer)  
->データ処理モジュール

層はデータから表現 (representaiton) を抽出

層をつなぎ合わせることで段階的なデータ蒸留(data distillation) を実装する

ディープラーニングモデル  
-> データ処理のふるいのようなもの

Dense層  
- 全結合層  
- ソフトマックス層

コンパイルステップ
- 損失関数
- オプティマイザ(更新メカニズム)
- メトリックス(制度などの指標)


テンソル(tensor)  
テンソルの次元(dimension) -> 軸(axis)

数値を1つしか含んでいないテンソル -> スカラー(scalar)

テンソルの軸の数-> 階数(rank)

```
x = np.array(12)
x.ndim > 0 #スカラーテンソルの軸の数は0
```

テンソルの重要な属性
- 軸の数(階数) #テンソルのndim属性
- 形状 #整数のタプル
- データ型 #pythonでは通常dtype

テンソル分解
```
my_slice.shape > (60000, 28, 28)
#同義
my_slice = train_images[10:100]
my_slice = train_images[10:100, :, :]
my_slice = train_images[10:100, 0:28, 0:28]
```

データバッチ
最初の軸(軸0) : バッチ軸(batch axis)

データテンソルの例
- ベクトルデータ : 形状が(samples, features)の2次元テンソル
- 時系列データ、シーケンスデータ : 形状が(samples, timesteps, features)の3次元テンソル
- 画像 : 形状が(samples, height, width, channels) の4次元テンソル
- 動画 : 形状が(samples, framges, height, width, channels) の5次元テンソル

channels -> 色(3)  
Theano > チャネルファースト  
Tensorflow > チャネルラスト

```
assert #false のときエラーログ
```
```
x = x.copy()
#浅いコピー : 元のオブジェクトのコンテンツへの参照をもつコンテナ
```

BLAS(Basic Linear Algebra Subprograms)

ブロードキャスト  
小さい方のテンソルが大きい方のテンソルに形を合わせる

転置
```
x = np.zeros((300, 20))
x = np.transpose(x)
print(x.shape) > (20, 300)
```
