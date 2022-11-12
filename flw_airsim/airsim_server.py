from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

# Define strategy
strategy = fl.server.strategy.FedAvg(min_fit_clients=1,
                                    min_evaluate_clients=1,
                                    min_available_clients=1,
                                    )

# Start Flower server
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
