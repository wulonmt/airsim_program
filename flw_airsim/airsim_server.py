from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

# Define strategy
strategy = fl.server.strategy.FedAdam(min_fit_clients=2,
                                    min_evaluate_clients=2,
                                    min_available_clients=2,
                                    )

# Start Flower server
fl.server.start_server(
    server_address="192.168.1.85:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)
