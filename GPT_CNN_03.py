import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 5つの100x100の2次元配列を用意
arrays = [np.random.rand(100, 100) for _ in range(5)]

# それぞれの2次元配列を1次元に変換
flattened_arrays = [array.flatten() for array in arrays]

# 1次元配列を連結して、入力データを作成
input_data = np.stack(flattened_arrays, axis=1)
print(input_data.shape) # shape: (10000, 5)

# ターゲットデータを生成（崩壊: 2%, 非崩壊: 98%）
num_samples = input_data.shape[0]
collapse_rate = 0.02

# ランダムに崩壊（1）と非崩壊（0）を設定
target_data = np.random.choice([0, 1], size=num_samples, p=[1-collapse_rate, collapse_rate])

# 崩壊と非崩壊のインデックスを取得
collapse_indices = np.where(target_data == 1)[0]
non_collapse_indices = np.where(target_data == 0)[0]

# 崩壊データの数
num_collapse = len(collapse_indices)

# 非崩壊データからランダムに5倍の数をサンプリング
num_samples = num_collapse * 5
sampled_non_collapse_indices = np.random.choice(non_collapse_indices, size=num_samples, replace=False)

# 崩壊データとサンプリングした非崩壊データを結合
balanced_indices = np.concatenate((collapse_indices, sampled_non_collapse_indices))

# サンプリングされたデータとターゲットを取得
X_balanced = input_data[balanced_indices]
y_balanced = target_data[balanced_indices]

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

print(f"Train data shape: {X_train.shape}, Train target shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}, Test target shape: {y_test.shape}")

# モデルの構築
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# モデルのコンパイル
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# モデルの訓練
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# モデルの評価
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
