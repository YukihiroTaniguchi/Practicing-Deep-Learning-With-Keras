### 181205(Wed)  
p.159 - p.166
chapter5.py : L407-L495
##### *Remember me*  
ファインチューニング  
-> 凍結された畳み込みベースの出力側の層をいくつか解凍し、モデルの新しく追加された部分と回答した層の両方で訓練を行う

ファインチューニングの手順
1. 訓練済みのベースネットワークの最後にカスタムネットワークを追加する
2. ベースネットワークを凍結する
3. 追加した部分の訓練を行う
4. ベースネットワークの一部の層を解凍する
5. 解凍した層と追加した部分の訓練を同時に行う

出力側の層のみをファインチューニングする
-> 理由は2点

* 入力側は汎用的で再利用可能な特徴量をエンコードしている  
-> 出力側はより具体的な特徴量をエンコードしている  
-> 入力側の層をファインチューニングすれば、収穫逓減が早まる
* 訓練対称のパラメータの数が増えるほど、過学習のリスクが高まる
-> なるべく少なくする

例 : block5_conv1以降の層のみ凍結を解除する
```python
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
```


結果のプロット
-> 正解率と損失値
```python
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
#正解率
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

#損失値
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()

plt.show()
```

プロットのスムージング  
データポイントの指数移動平均に置き換える
```python
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
```
