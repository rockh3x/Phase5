import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.server import ServerConfig
import matplotlib.pyplot as plt
import csv

round_accuracies = []
client_accuracy_log = {}
sent_bytes_log = []
client_id_map = {}

class LoggingFedAvg(FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        total_bytes = 0
        for client_id, res in results:
            sent = res.metrics.get("sent_bytes", 0)
            total_bytes += sent
        sent_bytes_log.append((rnd, total_bytes))
        print(f"[Round {rnd}] Total Data Sent: {total_bytes/1024:.2f} KB")
        return super().aggregate_fit(rnd, results, failures)

    def aggregate_evaluate(self, rnd, results, failures):
        client_accs = {}
        for client_obj, res in results:
            acc = res.metrics.get("accuracy")
            if acc is not None:
                if client_obj not in client_id_map:
                    client_id_map[client_obj] = f"Client {len(client_id_map)}"
                readable_id = client_id_map[client_obj]
                client_accs[readable_id] = acc

        if client_accs:
            avg_acc = sum(client_accs.values()) / len(client_accs)
            round_accuracies.append((rnd, avg_acc))
            print(f"[Round {rnd}] Avg Accuracy: {avg_acc:.4f}")
            for cid, acc in client_accs.items():
                print(f" - {cid}: {acc:.4f}")
                client_accuracy_log.setdefault(cid, []).append(acc)
        else:
            print(f"[Round {rnd}] No accuracy data.")

        return super().aggregate_evaluate(rnd, results, failures)

def plot_accuracy():
    if not round_accuracies:
        print("No accuracy data to plot.")
        return
    rounds, accs = zip(*round_accuracies)
    plt.figure(figsize=(8, 4))
    plt.plot(rounds, accs, marker='o', label='Global Accuracy')
    for cid, acc_list in client_accuracy_log.items():
        plt.plot(range(1, len(acc_list)+1), acc_list, label=cid)
    plt.title("Federated Accuracy per Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("federated_accuracy_final.png", dpi=300)
    plt.show()

def save_communication_log():
    with open("communication_cost.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Total_Sent_Bytes"])
        for rnd, sent in sent_bytes_log:
            writer.writerow([rnd, sent])

if __name__ == "__main__":
    strategy = LoggingFedAvg(
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
    )
    try:
        fl.server.start_server(
            server_address="localhost:8080",
            config=ServerConfig(num_rounds=5),
            strategy=strategy,
        )
    finally:
        plot_accuracy()
        save_communication_log()
