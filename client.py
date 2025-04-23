import flwr as fl
import sys
import pickle
from tensorflow.keras.datasets import mnist
from utils import create_non_iid
from model import build_model

client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
partitions = create_non_iid(num_clients=5)
x_train, y_train = partitions[client_id]
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

model = build_model()

def get_model_size(weights):
    return len(pickle.dumps(weights))

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        size = get_model_size(model.get_weights())
        print(f"📤 Client {client_id} sent ~{size/1024:.2f} KB of weights")
        return model.get_weights(), len(x_train), {"sent_bytes": size}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())
