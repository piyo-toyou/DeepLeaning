import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

# データセットの作成
height, width = 300, 300

# 1～5層のデータを作成
layer_1 = np.random.rand(height, width)  # 標高値
layer_2 = np.random.rand(height, width)  # 傾斜量
layer_3 = np.random.rand(height, width)  # 降水量
layer_4 = np.random.rand(height, width)  # 植生分布
layer_5 = np.random.rand(height, width)  # 地質情報

# 6層目のデータを作成（崩壊分布、2%の確率で崩壊）
layer_6 = np.random.choice([0, 1], size=(height, width), p=[0.98, 0.02])

# データをスタック
input_data = np.stack([layer_1, layer_2, layer_3, layer_4, layer_5], axis=-1)
target_data = layer_6

# データの形状が正しいことを確認
print(input_data.shape)  # (300, 300, 5)
print(target_data.shape)  # (300, 300)

# 崩壊箇所を抽出
collapse_indices = np.where(target_data == 1)
non_collapse_indices = np.where(target_data == 0)

# 非崩壊箇所のランダムサンプリング
num_collapse = len(collapse_indices[0])
non_collapse_indices_sampled = np.random.choice((non_collapse_indices[0]), size=num_collapse*5, replace=False)

# サンプルデータの構築
sampled_indices = (np.concatenate((collapse_indices[0], non_collapse_indices[0][non_collapse_indices_sampled])),
                   np.concatenate((collapse_indices[1], non_collapse_indices[1][non_collapse_indices_sampled])),
                   np.concatenate((collapse_indices[2], non_collapse_indices[2][non_collapse_indices_sampled])))

sampled_input_data = input_data[sampled_indices]
sampled_target_data = target_data[sampled_indices]

# トレーニングとテストの分割
X_train, X_test, y_train, y_test = train_test_split(sampled_input_data, sampled_target_data, test_size=0.2, random_state=42)

# トレーニングデータの形状が正しいことを確認
print(X_train.dtype)
print(y_train.dtype)
print(X_train.shape)
print(y_train.shape)

# モデルの構築
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, 5)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # 崩壊確率を出力
])

# モデルのコンパイル
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# モデルのトレーニング
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# モデルの評価
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
