import numpy as np
from tensorflow.keras.datasets import mnist

def create_non_iid(num_clients=5):
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    partitions = []
    sizes = [6000, 3000, 1000]
    for i in range(num_clients):
        labels = [(2 * i) % 10, (2 * i + 1) % 10]
        idx = np.where(np.isin(y_train, labels))[0]
        np.random.shuffle(idx)
        size = sizes[i % len(sizes)]
        partitions.append((x_train[idx][:size], y_train[idx][:size]))
    return partitions
