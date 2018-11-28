### 181128(Wed)  
p.84 - p.94
chapter3.py : L227-L405
##### *Remember me*  
```python
results = model.evaluate(x_test, one_hot_test_labels)
# バッチごとにある入力データにおける損失値を計算
```
```python
predictions = model.predict(x_test)
# 入力サンプルに対する予測値の出力を生成
```

ラベルをエンコードする方法  
->ラベルを整数のテンソルとしてキャストしてもよい
```python
y_train = np.array(train_labels)
y_test = np.array(test_labels)

model.compile(optimizer = 'rmsprop',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['acc'])
              # ラベルが整数の場合は損失関数としてsparse_categorical_crossentropyを使用
              # categorical_crossentropyと数学的な違いはない
```
多クラス分類(N個) :  
最後の層はサイズがNのDense層、活性化関数としてsoftmaxを使用すべき  
->出力はN個の確率分布になる  
損失関数 : categorical_crossentropy(他クラス交差エントロピー)  
one_hotエンコーディング : categorical_crossentropy  
整数エンコーディング : sparse_categorical_crossentropy  
分類先のカテゴリ数が多い場合は中間層が小さすぎることで情報ボトルネックが生じないようにする

回帰
データの正規化  
特徴量の平均値を引き、標準偏差で割る
```python
mean = train_data.mean(axis=0) #平均
train_data -= mean
std = train_data.std(axis=0) #標準偏差
train_data /= std

test_data -= mean
test_data /= std
```
正規化でもテストデータを使用してはならない

回帰問題の損失関数  
-> mse(平均二乗誤差)
回帰問題の監視指標
-> mae (平均絶対誤差)

データポイントが少ない場合  
->検証は k分割交差検証  
```python
k = 4
num_val_samples = len(train_data) // k #切り捨て除算
num_epochs = 100
all_scores = []
for i in range(k):
    print('prosessing fold #', i)
    val_data = \
        train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = \
        train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        #検証データの両脇のデータを足す->訓練データ
    partial_train_data = np.concatenate( #配列の結合
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
         axis=0)

    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
         axis=0)

    model = build_model()

    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
```
