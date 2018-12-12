### 181212(Wed)    
p.241 - p.357
chapter6.py : L200 - L205
##### *Remember me*  
シーケンス（= データセット中でランクi（1から始まる）の単語がインデックスiを持つ単語インデックスのリスト）  
-> 文の順番にインデックスがついてる

```python
#Kerasを使った単語レベルでのone-hotエンコーディング
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

#出現頻度の高い1000この単語だけを処理する設定
tokenizer = Tokenizer(num_words=1000)

#単語にインデックスをつける
tokenizer.fit_on_texts(samples)

#文字列を整数インデックスのリストに変換
sequences = tokenizer.texts_to_sequences(samples)
sequences #[[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]

#文字列をone-hotエンコーディングすることも可能
kone_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
one_hot_results.shape #(2, 1000)
```

学習済みの特徴量の埋め込みをモデルに読み込む
```python
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#準備した単語埋め込みをEmbeddingに読み込む
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
#せっかく覚えた重みを忘れないように凍結する
```
