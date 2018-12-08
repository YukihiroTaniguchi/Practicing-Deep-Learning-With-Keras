### 181208(Sat)  
p.182 - p.186
chapter5.py : L689-L740
##### *Remember me*  
クラス活性化をヒートマップとして可視化  

Grad-CAM  
「入力画像によって様々なチャネルがどれくらい強く活性化されるか」を表す空間マップ  
\*  
「そのクラスにとって各チャネルがどれくらい重要か」を表す値で重み付けする  
->  
「入力画像によってそのクラスがどれくらい強く活性化されるか」を表す空間マップが得られる  

VGG16モデルに合わせて入力画像を前処理
```python
model = VGG16(weights='imagenet')
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

img_path = '/Users/yukihiro/Documents/Practicing-Deep-Learning-With-Keras/others/creative_commons_elephant.jpg'

#ターゲット画像を読み込む : imgはサイズが224 × 224のPIL画像
img = image.load_img(img_path, target_size=(224, 224))

#xは形状が(224, 224, 3)のfloat32型のNumPy配列
x = image.img_to_array(img)

#(1, 224, 224, 3)のバッチに変換するために次元を追加
x = np.expand_dims(x, axis=0)

#バッチの前処理(チャネルごとに色を正規化)
x = preprocess_input(x)
```

Grad-CAM
```python
preds = model.predict(x)
afrtican_elepaht_output = model.output[:, 386]

#VGG16の最後の畳み込み層であるblock5_conv3の出力特徴マップ
last_conv_layer = model.get_layer('block5_conv3')

#block5_conv3の出力特徴マップでの「アフリカゾウ」クラスの勾配
grads = K.gradients(afrtican_elepaht_output, last_conv_layer.output)[0]

#形状が(512, )のベクトル
#各エントリは特定の特徴マップチャネルの勾配の平均強度
pooled_grads = K.mean(grads, axis=(0, 1, 2))

#2頭のアフリカゾウのサンプル画像に基づいて、pooled_gradsと
#block5_conv3の出力特徴マップの値にアクセスするための関数
iterate = K.function([model.input],
                     [pooled_grads, last_conv_layer.output[0]])

#これら2つの値をNumPy配列として取得
pooled_grads_value, conv_layer_output_value = iterate([x])

#「アフリカゾウ」クラスに関する「このチャネルの重要度」を
#特徴マップ配列の各チャネルに掛ける
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

#最終的な特徴マップのチャネルごとの平均値が
#クラスの活性化のヒートマップ
heatmap = np.mean(conv_layer_output_value, axis=-1)

#ヒートマップの後処理
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
import matplotlib.pyplot as plt
plt.matshow(heatmap)
```

元の画像にスーパーインポーズ
```python
import cv2

#cv2を使って元の画像を読み込む
img = cv2.imread(img_path)

#もとの画像と同じサイズになるようにヒートマップのサイズを変更
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

#ヒートマップをRGBに変換
heatmap = np.uint8(255 * heatmap)

#ヒートマップをもとの画像に適用
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#0.4はヒートマップの強度係数
superimposed_img = heatmap * 0.4 + img

#保存
cv2.imwrite('/Users/yukihiro/Documents/Practicing-Deep-Learning-With-Keras/others/elephant_cam.jpg', superimposed_img)
```
