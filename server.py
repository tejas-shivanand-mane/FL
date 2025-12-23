import flwr as fl
from typing import Dict, List, Tuple
import logging
import argparse
import csv
import os
from datetime import datetime


# Logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FL-Server")


# Results file

RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "fl_results.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)


# Metric aggregation

def weighted_average(
    metrics: List[Tuple[int, Dict[str, float]]]
) -> Dict[str, float]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    for num_examples, client_metrics in metrics:
        for k, v in client_metrics.items():
            aggregated[k] = aggregated.get(k, 0.0) + v * num_examples

    for k in aggregated:
        aggregated[k] /= total_examples

    return aggregated


# Strategy mixin (MRO-safe)

class LoggingStrategyMixin:
    def __init__(self, strategy_name: str, *args, **kwargs):
        self.strategy_name = strategy_name
        self.final_metrics = None
        super().__init__(*args, **kwargs)

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        if aggregated is not None:
            loss, metrics = aggregated
            logger.info(
                f"📊 Round {server_round} — loss={loss:.4f}, metrics={metrics}"
            )
            self.final_metrics = (loss, metrics)

        return aggregated

    def write_final_results(self, num_rounds: int):
        if self.final_metrics is None:
            return

        loss, metrics = self.final_metrics
        write_header = not os.path.exists(RESULTS_FILE)

        with open(RESULTS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                        "timestamp",
                        "strategy",
                        "num_rounds",
                        "final_loss",
                        *metrics.keys(),]
                )

            writer.writerow(
                [
                    datetime.now().isoformat(),
                    self.strategy_name,
                    num_rounds,
                    loss,
                    *metrics.values(),
                ]
            )


# Strategy implementations

class LoggingFedAvg(LoggingStrategyMixin, fl.server.strategy.FedAvg):
    pass

class LoggingFedProx(LoggingStrategyMixin, fl.server.strategy.FedProx):
    pass


# Strategy factory

def create_strategy(name: str):
    common_args = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    if name == "fedavg":
        return LoggingFedAvg(
            strategy_name="FedAvg",
            **common_args,
        )

    if name == "fedprox":
        return LoggingFedProx(
            strategy_name="FedProx",
            proximal_mu=0.01,
            **common_args,
        )

    raise ValueError(f"Unknown strategy: {name}")

# Server main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        default="fedprox",
        choices=["fedavg", "fedprox"],
    )
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()

    strategy = create_strategy(args.strategy)

    logger.info(f"🚀 Starting Flower server with {args.strategy}")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    strategy.write_final_results(args.rounds)
    logger.info("✅ Final results written to file")

if __name__ == "__main__":
    main()
