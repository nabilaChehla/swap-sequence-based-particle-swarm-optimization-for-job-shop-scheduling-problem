import os
from src.modules.jsspProcessor import JSSPProcessor

if __name__ == "__main__":
    dataset_folder = "./src/data/processed/data_20j_15m"

    for filename in os.listdir(dataset_folder):
        if filename.endswith(".txt"):
            dataset_path = os.path.join(dataset_folder, filename)
            processor = JSSPProcessor(dataset_path)
            best_schedule, best_makespan, exec_time = processor.run_pso(
                num_particles=200,
                max_iter=200,
                w=0.7,
                c1=0.8,
                c2=1,
                adaptive_params=True,
                mutation_rate=1,
                max_stagnation=20,
                early_stopping_window=None,
                improvement_threshold=0.01,
            )

            """
# Max stagnation less -> better (from 50 to 20 ) improved
# w less -> better (from 0.8 to 0.5 ) improved
# Max stagnation less -> (from 20 to 10 ) same result


good param but takes too long


            best_schedule, best_makespan, exec_time = processor.run(
                num_particles=80,
                max_iter=1000,
                w=0.4,
                c1=0.8,
                c2=0.5,
                adaptive_params=False,
                mutation_rate=1,
                max_stagnation=20,
                early_stopping_window=None,
                improvement_threshold=0.01,
            )




import os
from src.modules.jsspProcessor import JSSPProcessor

if __name__ == "__main__":
    dataset_folder = "./src/data/processed/data_20j_15m"

    for filename in os.listdir(dataset_folder):
        if filename.endswith(".txt"):
            dataset_path = os.path.join(dataset_folder, filename)
            processor = JSSPProcessor(dataset_path)
            best_schedule, best_makespan, exec_time = processor.run_ga(
                population_size=50,
                max_iter=200,
                crossover_rate=0.9,
                mutation_rate=1,
                tournament_size=3,
                max_stagnation=20,
                early_stopping_window=None,
                improvement_threshold=0.001,
            )

import os
from src.modules.jsspProcessor import JSSPProcessor

if __name__ == "__main__":
    dataset_folder = "./src/data/processed/data_20j_15m"

    for filename in os.listdir(dataset_folder):
        if filename.endswith(".txt"):
            dataset_path = os.path.join(dataset_folder, filename)
            processor = JSSPProcessor(dataset_path)
            best_schedule, best_makespan, exec_time = processor.run_priority_pso(
                num_particles=100,
                max_iter=1000,
                w=0.9,
                c1=2.0,
                c2=1.8,
                mutation_rate=0.3,
                adaptive_params=True,
                max_stagnation=20,
                early_stopping_window=100,
                improvement_threshold=0.01,
                apply_local_search=True,
                verbose=True,
            )

            
            """
