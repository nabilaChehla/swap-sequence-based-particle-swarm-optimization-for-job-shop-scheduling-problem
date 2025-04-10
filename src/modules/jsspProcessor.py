from src.modules.pso import PSOOptimizer
from src.modules.datasetParser import DatasetParser
import os
import time
import csv
from src.modules.modelisation import JSSP
from src.modules.visualisation import ScheduleVisualizer
from src.modules.genetic import GAOptimizer  # This would be your new GA optimizer class
from src.modules.PSPO import PriorityParticle


class JSSPProcessor:
    def __init__(
        self,
        dataset_path,
        plot: bool = True,
        output_base="output",
    ):
        self.dataset_path = dataset_path
        self.dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.output_dir = os.path.join(output_base, self.dataset_name)
        self.log_file = os.path.join(self.output_dir, "results.csv")
        self.plot = plot

    def run_priority_pso(
        self,
        num_particles: int = 50,
        max_iter: int = 500,
        w: float = 0.7,
        c1: float = 1.7,
        c2: float = 1.7,
        mutation_rate: float = 0.3,
        adaptive_params: bool = True,
        max_stagnation: int = 30,
        early_stopping_window: int = 50,
        improvement_threshold: float = 0.005,
        apply_local_search: bool = True,
        verbose: bool = True,
    ):
        """Run Priority-based PSO optimization"""
        with open(self.dataset_path, "r") as file:
            dataset_str = file.read()

        num_jobs, num_machines, upper_bound, lower_bound, times, machines = (
            DatasetParser.parse(dataset_str)
        )
        os.makedirs(self.output_dir, exist_ok=True)

        jssp = JSSP(machines, times)

        optimizer = PriorityParticle(jssp)
        start_time = time.time()
        best_schedule, best_makespan = optimizer.optimize(
            num_particles=num_particles,
            max_iter=max_iter,
            w=w,
            c1=c1,
            c2=c2,
            mutation_rate=mutation_rate,
            adaptive_params=adaptive_params,
            max_stagnation=max_stagnation,
            early_stopping_window=early_stopping_window,
            improvement_threshold=improvement_threshold,
            apply_local_search=apply_local_search,
            verbose=verbose,
        )
        exec_time = time.time() - start_time

        self._process_results(jssp, optimizer, best_makespan, exec_time, "PriorityPSO")
        return best_schedule, best_makespan, exec_time

    def run_pso(
        self,
        num_particles: int = 30,
        max_iter: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        adaptive_params: bool = True,
        mutation_rate: float = 0.1,
        max_stagnation: int = 15,
        early_stopping_window: int = 20,
        improvement_threshold: float = 0.01,
    ):
        """Run PSO optimization"""
        with open(self.dataset_path, "r") as file:
            dataset_str = file.read()

        num_jobs, num_machines, upper_bound, lower_bound, times, machines = (
            DatasetParser.parse(dataset_str)
        )
        os.makedirs(self.output_dir, exist_ok=True)

        jssp = JSSP(machines, times)

        optimizer = PSOOptimizer(jssp)
        start_time = time.time()
        best_schedule, best_makespan = optimizer.optimize(
            num_particles=num_particles,
            max_iter=max_iter,
            w=w,
            c1=c1,
            c2=c2,
            adaptive_params=adaptive_params,
            mutation_rate=mutation_rate,
            max_stagnation=max_stagnation,
            early_stopping_window=early_stopping_window,
            improvement_threshold=improvement_threshold,
        )
        exec_time = time.time() - start_time

        self._process_results(jssp, optimizer, best_makespan, exec_time, "PSO")
        return best_schedule, best_makespan, exec_time

    def run_ga(
        self,
        population_size: int = 50,
        max_iter: int = 200,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        tournament_size: int = 3,
        max_stagnation: int = 20,
        early_stopping_window: int = 30,
        improvement_threshold: float = 0.005,
    ):
        """Run GA optimization"""
        with open(self.dataset_path, "r") as file:
            dataset_str = file.read()

        num_jobs, num_machines, upper_bound, lower_bound, times, machines = (
            DatasetParser.parse(dataset_str)
        )
        os.makedirs(self.output_dir, exist_ok=True)

        jssp = JSSP(machines, times)

        optimizer = GAOptimizer(jssp)
        start_time = time.time()
        best_schedule, best_makespan = optimizer.optimize(
            population_size=population_size,
            max_iter=max_iter,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            max_stagnation=max_stagnation,
            early_stopping_window=early_stopping_window,
            improvement_threshold=improvement_threshold,
        )
        exec_time = time.time() - start_time

        self._process_results(jssp, optimizer, best_makespan, exec_time, "GA")
        return best_schedule, best_makespan, exec_time

    def _process_results(self, jssp, optimizer, best_makespan, exec_time, algorithm):
        """Common result processing for both algorithms"""
        if self.plot:
            # Create algorithm-specific subfolder
            algo_dir = os.path.join(self.output_dir, algorithm.lower())
            os.makedirs(algo_dir, exist_ok=True)

            ScheduleVisualizer.plot_convergence(
                optimizer.iteration_history,
                optimizer.makespan_history,
                save_folder=algo_dir,
                title=f"{algorithm} Convergence",
            )
            ScheduleVisualizer.plot_gantt_chart(
                jssp, save_folder=algo_dir, title=f"{algorithm} Schedule"
            )

            self._log_results(best_makespan, exec_time, algorithm)

        print(
            f"[{self.dataset_name} - {algorithm}] "
            f"Makespan: {best_makespan} | "
            f"Time: {exec_time:.2f}s"
        )

    def _log_results(self, best_makespan, exec_time, algorithm):
        """Log results with algorithm information"""
        csv_exists = os.path.exists(self.log_file)
        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not csv_exists:
                writer.writerow(
                    ["Dataset", "Algorithm", "Best Makespan", "Execution Time (s)"]
                )
            writer.writerow(
                [self.dataset_name, algorithm, best_makespan, f"{exec_time:.4f}"]
            )
