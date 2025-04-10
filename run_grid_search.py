from src.modules.gridSearch import PSOGridSearch


def main():
    # Configuration
    dataset_file = "src/data/processed/data_20j_15m/data_20j_15m_1.txt"  # Folder containing your JSSP instance files
    output_params = (
        "SPSO/paramaters/best_params_grid.json"  # Output file for best parameters
    )
    output_history = (
        "SPSO/paramaters/search_history_grid.csv"  # Output file for search history
    )

    # Initialize and run grid search
    grid_search = PSOGridSearch(
        dataset_file=dataset_file,
        params_output_file=output_params,
        history_output_file=output_history,
    )

    # Optional: Customize parameter grid if needed
    grid_search.set_parameter_grid(
        {
            "num_particles": [20, 30, 50, 100],  # Specific values
            "max_iter": [100, 200, 500],
            "w": [0.3, 0.5, 0.8, 0.9],  # Range
            "c1": [0.5, 0.75, 1],  # Fixed value
            "c2": [0.5, 0.75, 1],  # Fixed value
            "mutation_rate": [1],  # Fixed
            "max_stagnation": [20],
            "early_stopping_window": [None],  # Fixed
            "improvement_threshold": [0.01],
        },
    )

    # Optional: Customize parameter grid if needed

    print("Starting grid search...")
    results = grid_search.run_search()

    print("\nGrid search completed!")
    print(f"Best parameters saved to: {output_params}")
    print(f"Search history saved to: {output_history}")
    print(f"Best average makespan: {results['best_avg_makespan']:.2f}")


if __name__ == "__main__":
    main()
