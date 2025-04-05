import numpy
from src.modules.modelisation import JSSP
from src.modules.visualisation import ScheduleVisualizer
from src.modules.pso import PSOOptimizer


# Example usage
if __name__ == "__main__":
    machines_matrix = [
        [1, 2, 3],  # Job 0 operations
        [1, 3, 2],  # Job 1 operations
    ]

    times_matrix = [
        [5, 10, 8],  # Job 0 processing times
        [7, 9, 6],  # Job 1 processing times
    ]

    # Create and run optimization
    jssp = JSSP(machines_matrix, times_matrix)
    optimizer = PSOOptimizer(jssp)
    best_schedule, best_makespan = optimizer.optimize(num_particles=20, max_iter=2)

    # Display results
    print(best_schedule)

    # Visualize results
    jssp.evaluate_schedule(best_schedule)
    ScheduleVisualizer.plot_convergence(
        optimizer.iteration_history, optimizer.makespan_history, save_folder="./test"
    )
    ScheduleVisualizer.plot_gantt_chart(jssp, save_folder="./test")
