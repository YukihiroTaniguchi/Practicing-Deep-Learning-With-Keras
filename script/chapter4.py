#4.2.1 訓練データセット、検証データセット、テストデータセット
#ホールドアウト法
nun_validation_samples = 10000
np.random.shuffle(data)
validation_data = data[:num_validation_samples]
data = data[num_validation_samples:]
training_data = data[:]
model = get_model()
model.train(training_data)
validataion_score = model.evaluate(validation_data)
test_score = model.evaluate(test_data)

#k分割交差検証
k = 4
num_validation_samples = len(data) // k
np.random.shuffle(data)
validataion_scores = []

for fold in range(k):
    validation_data = data[num_validation_samples * fold:
                           num_validation_samples * (fold + 1)]
    training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]

    model = get_model()
    model.train(training_data)

    validation_score = model.evaluate(validation_data)
    validation_scores.append(validation_score)

validation_score = np.average(validation_scores)
model = get_model()
model.train(data)
test_score = model.evaluate(test_data)

#4.4.1 ネットワークのサイズを削減する
from keras import models
form keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#小さくする
model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#大きくする
model =models.Sequential()
model.add(layers.Dense(512, activaion='relu', input_shape=(10000,)))
model.add(layers.Dense(512, activaion='relu'))
model.add(layers.Dense(1, activaion='sigmoid'))

#4.4.2 重みを正則化する
#モデルにL2正則化を追加
from keras import regularizersomodel = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001),
                       activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer = regularizers.l2(0.001),
                       activation='relu', input_shape=(10000,)))
model.add(layers.Dense(1, activation='sigmoid'))

#kerasの様々な正則化項
from keras import regularizers
#L1正則化
regularizers.l1(0.001)
regularizers.l1_l2(l1=0.001, l2=0.001)

#4.4.3 ドロップアウト
model = models.Sequential()
model.add(layers.Dense(16, activaion='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activaion='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
