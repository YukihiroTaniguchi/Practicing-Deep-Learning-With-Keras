### 181121(Wen)  
p.28 - p.31  
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
