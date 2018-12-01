#### chapter3 まとめ  
教師あり学習  

###### 1. 二値分類  
IMDbデータセット  
映画の「肯定的」または「否定的な」50,000件のレビューで構成されている  

データの前処理
```python
train_data[0]
#[1, 14, 22, 16, ... 178, 32]
train_label[0]
#1
#one-hot エンコーディングによって0と1のベクトルに変換
def vectorize_sequences(sequences, dimension=10000):

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.

    return results
x_train = vectorize_sequences(train_data)
x_train[0]
#array([0., 1., 1., ..., 0., 0., 0.])
#もとのtrain_dataの配列に2が入っていたとしたら、index 2が1に変換される
```
活性化関数 : ReLU  
最後の層の活性化関数 : Sigmoid 単一ユニット  
損失関数 : binary_crossentropy
評価指標 : 正解率(accuracy)  
オプティマイザ : rmsprop

###### 2. 他クラス分類
Reuters データセット  
46種類のトピックをもつニュース記事のデータセット


データの前処理  
IMDbデータセットと同じ  

ラベルのベクトル化  
one-hotエンコーディング  
->カテゴリエンコーディングと呼ばれる  
```python
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_train_labels = to_one_hot(train_labels)
#下記は同様の動作をするkerasの組み込み関数
from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_train_labels = to_categorical(test_labels)
```
活性化関数 : ReLU  
最後の層の活性化関数 : Sigmoid ラベル数のユニット(確率分布->合計すると1)  
損失関数 : categorical_crossentropy  
- one-hot : categorical_crossentropy  
- 整数 : sparse_categorical_cross_entropy
評価指標 : 正解率(accuracy)  
オプティマイザ : rmsprop

###### 3. スカラー回帰
Boston Housing データセット  
特徴量はそれぞれ異なる尺度(犯罪発生率、地方財産税など)
ボストン近郊での住宅価格の中央値を予測する

データの前処理  
特徴量ごとの正規化  
特徴量の平均値を引き、標準偏差で割る  
-> 特徴量の中心が0になり、標準偏差が1になる
```python
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
```
※正規化もテストデータを使ってはならない  
活性化関数 : ReLU  
最後の層の活性化関数 : 単一ユニット(活性化関数はいらない)  
損失関数 : 平均二乗誤差(mse)  
評価指標 : 平均絶対誤差(mae)  
オプティマイザ : rmsprop  

###### 利用可能なデータが少ない場合
k分割交差検証
