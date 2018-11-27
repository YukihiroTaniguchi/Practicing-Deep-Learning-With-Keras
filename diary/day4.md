### 181127(Tue)  
p.65 - p.78  
chapter3.py : L32-L133
##### *Remember me*  
IMDbデータセット  
訓練用25000 : テスト用25000  
それぞれ否定的レビュー50% : 肯定的レビュー50%  
```python
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# numwords = 10000 -> 出現頻度が最も高い10000wordsだけ残し、他を捨てる
```
```python
max([max(sequence) for sequence in train_data])
# 最大のindexを調べる
```

モデルを訓練する際の正解率を訓練とは別のデータで行う
```python
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
```

model.fitで学習後、historyオブジェクトを返す  
->historyオブジェクトにはhistoryというメンバーがある  
->訓練中、検証中に監視される指標ごとにエントリが含まれている
```python
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_val, y_val))

history_dict = history.history
history_dict.keys()
#dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
```

前処理の重要性  

二値分類問題では、ネットワークの最後の層は、活性化関数のシグモイドを使用するDense層になる  
ネットワークの出力は、確率を表す0 ~ 1 のスカラー値になる  
出力がシグモイドのスカラー値である場合には、損失関数としてbinary_crossentropyを使用すべきである  
rmspropオプティマイザはどの問題でも十分に良い選択となる

他クラス単一ラベル分類

多いラベルのベクトル化  
-> one-hot エンコーディング  
-> カテゴリエンコーディングともいう

```python
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_train_labels = to_one_hot(train_labels)
#ラベルのインデックスの位置に1が含まれている以外はすべて0が設定されたベクトルにする
###################################################################################
from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_train_labels = to_categorical(test_labels)
#上のkeras組み込み関数で同じことができる
```

他クラス単一ラベル分類  
出力はクラス数だけある  
確率分布  
46クラスの場合、46次元の出力ベクトルをsoftmaxで生成し、合計は1となる  
->categorical_crossentropyが最適  
->確率分布とラベルの真の分布との距離になる  
->これを最小化する
