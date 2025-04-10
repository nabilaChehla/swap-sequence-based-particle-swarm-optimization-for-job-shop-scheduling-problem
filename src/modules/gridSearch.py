import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Tuple, Union
from itertools import product
from src.modules.jsspProcessor import JSSPProcessor


class PSOGridSearch:
    def __init__(
        self,
        dataset_file: str,
        params_output_file: str = "best_params.json",
        history_output_file: str = "search_history_grid.csv",
    ):
        self.dataset_file = dataset_file
        self.params_output_file = params_output_file
        self.history_output_file = history_output_file
        self.best_results: Dict[str, Any] = {}

        # Define fixed fieldnames for CSV
        self.fieldnames = [
            "timestamp",
            "parameters",
            "makespan",
            "exec_time",
            "num_particles",
            "max_iter",
            "w",
            "c1",
            "c2",
            "mutation_rate",
            "max_stagnation",
            "early_stopping_window",
            "improvement_threshold",
            "is_best",
        ]

        # Initialize CSV file with header
        self._initialize_csv_file()

        # Parameter grid
        self.param_grid = {
            "num_particles": [20, 30, 50, 100],
            "max_iter": [200, 500, 1000],
            "w": [0.3, 0.5, 0.8, 0.9],
            "c1": [0.5, 1, 1.5, 2],
            "c2": [0.5, 1, 1.5, 2],
            "mutation_rate": [1],
            "max_stagnation": [20],
            "early_stopping_window": [None],
            "improvement_threshold": [0.01],
        }

        self._validate_dataset_file()

    def _validate_dataset_file(self) -> None:
        if not os.path.isfile(self.dataset_file):
            raise ValueError(f"Dataset file not found: {self.dataset_file}")

    def _initialize_csv_file(self) -> None:
        """Initialize the CSV file with proper headers."""
        with open(self.history_output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def set_parameter_grid(self, grid: Dict[str, Union[List[Any], Any]]) -> None:
        self.param_grid = grid

    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        sampled_params = {}
        for param_name, param_values in self.param_grid.items():
            sampled_params[param_name] = (
                list(param_values)
                if isinstance(param_values, (list, tuple))
                else [param_values]
            )

        param_names = sampled_params.keys()
        value_combinations = product(*sampled_params.values())
        return [dict(zip(param_names, combo)) for combo in value_combinations]

    def evaluate_parameter_set(self, params: Dict[str, Any]) -> Tuple[float, float]:
        processor = JSSPProcessor(dataset_path=self.dataset_file, plot=False)
        _, makespan, exec_time = processor.run_pso(**params)
        return makespan, exec_time

    def run_search(self) -> Dict[str, Any]:
        param_combinations = self.generate_parameter_combinations()
        total_combinations = len(param_combinations)
        print(f"Starting grid search with {total_combinations} parameter combinations")

        best_params = None
        best_makespan = float("inf")
        best_exec_time = float("inf")

        for i, params in enumerate(param_combinations, 1):
            print(f"\nEvaluating combination {i}/{total_combinations}: {params}")
            makespan, exec_time = self.evaluate_parameter_set(params)

            # Determine if this is the new best
            is_best = False
            if makespan < best_makespan or (
                makespan == best_makespan and exec_time < best_exec_time
            ):
                best_params = params
                best_makespan = makespan
                best_exec_time = exec_time
                is_best = True

            # Record results immediately with consistent formatting
            self._record_search_result(params, makespan, exec_time, is_best)

            # Update best params file if this is the new best
            if is_best:
                self._save_current_best_params(
                    best_params, best_makespan, best_exec_time
                )

        return self._compile_final_results(best_params, best_makespan, best_exec_time)

    def _record_search_result(
        self,
        params: Dict[str, Any],
        makespan: float,
        exec_time: float,
        is_best: bool,
    ) -> None:
        """Record a single evaluation result with consistent CSV formatting."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "makespan": makespan,
            "exec_time": exec_time,
            "is_best": is_best,
        }

        # Add all parameters as separate columns
        record.update(params)

        # Convert None to empty string for CSV
        record = {k: v if v is not None else "" for k, v in record.items()}

        try:
            with open(self.history_output_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(record)
        except Exception as e:
            print(f"Error recording search result: {str(e)}")

    def _save_current_best_params(
        self,
        best_params: Dict[str, Any],
        best_makespan: float,
        best_exec_time: float,
    ) -> None:
        result = {
            "best_parameters": best_params,
            "makespan": best_makespan,
            "execution_time": best_exec_time,
            "timestamp": datetime.now().isoformat(),
            "dataset_file": self.dataset_file,
        }

        try:
            with open(self.params_output_file, "w") as f:
                json.dump(result, f, indent=4)
        except Exception as e:
            print(f"Error saving best parameters: {str(e)}")

    def _compile_final_results(
        self,
        best_params: Dict[str, Any],
        best_makespan: float,
        best_exec_time: float,
    ) -> Dict[str, Any]:
        return {
            "best_parameters": best_params,
            "makespan": best_makespan,
            "execution_time": best_exec_time,
            "search_history_file": self.history_output_file,
            "parameters_file": self.params_output_file,
            "dataset_file": self.dataset_file,
            "timestamp": datetime.now().isoformat(),
        }
