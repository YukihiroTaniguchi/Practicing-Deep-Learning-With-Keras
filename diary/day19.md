### 181219(Wed)    
p.262 - p.267
chapter7.py : L292 - L433
##### *Remember me*  
アイデア  
->  
実験  
->  
結果  
のループが開発

結果からアイデアを出すには？  
TensorBoard
- 訓練中に指標を視覚的に監視
- モデルのアーキテクチャの可視化
- 活性化と勾配のヒストグラムの可視化
- 埋め込みと勾配のヒストグラムの可視化
- 埋め込みを3次元で調査

TensorBoardコールバックを使ってモデルを訓練
```python
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='others/my_log_dir',     # ログファイルはこの場所に書き込まれる
        histogram_freq=1,                # 1エポックごとに活性化ヒストグラムを記録
        embeddings_freq=1,               # 1エポックごとに埋め込みデータを記録
        embeddings_layer_names=['embed'],
    )
]

history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=callbacks)
```
シェルで  
``` $ tensorboard --logdir=others/my_log_dir ```
