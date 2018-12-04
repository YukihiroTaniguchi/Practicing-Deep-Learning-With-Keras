### 181204(Tue)  
p.151 - p. 154
chapter5.py : L267-L311
##### *Remember me*  
学習済みのCNNを使用する
VGG16のインスタンス化
```python
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', #重みのチェックポイントを指定
                  include_top=False, #ネットワークの出力側にある全結合分類機を含めるかどうか
                  input_shape=(150, 150, 3)) #ネットワークに供給する画像テンソルの形状
```
学習済みの畳み込みベースを使って特徴量を抽出
```python
conv_base.summary()
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
pwd
base_dir = '/Users/hoge/Documents/Practicing-Deep-Learning-With-Keras/data/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory,
                                            target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break

    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)
```
