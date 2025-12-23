import flwr as fl
from typing import Dict, List, Tuple
import numpy as np
import logging

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FL-Server")

# -----------------------------
# Metric aggregation
# -----------------------------
def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate metrics from clients using weighted average."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated = {}
    for num_examples, client_metrics in metrics:
        for k, v in client_metrics.items():
            aggregated[k] = aggregated.get(k, 0.0) + v * num_examples

    for k in aggregated:
        aggregated[k] /= total_examples

    return aggregated

# -----------------------------
# Custom Strategy (FedProx)
# -----------------------------
class LoggingFedProx(fl.server.strategy.FedProx):
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        logger.info(f"🔄 Aggregating round {server_round}")
        logger.info(f"✔️  Successful clients: {len(results)}")
        logger.info(f"❌ Failed clients: {len(failures)}")

        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if aggregated_metrics is not None:
            loss, metrics = aggregated_metrics
            logger.info(
                f"📊 Round {server_round} — "
                f"loss={loss:.4f}, metrics={metrics}"
            )

        return aggregated_metrics

# -----------------------------
# Server main
# -----------------------------
def main() -> None:
    strategy = LoggingFedProx(
    proximal_mu=0.01,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
    )


    logger.info("🚀 Starting Flower server with FedProx")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
