from copy import deepcopy
from typing import List, Tuple, Dict
import random
import math
from operator import attrgetter
from itertools import chain
from collections import defaultdict


class Chromosome:
    """Represents a chromosome in the genetic algorithm population."""

    def __init__(
        self, sequence: List[Tuple[int, int]], job_machine_dict: Dict[int, List[int]]
    ):
        self.sequence = sequence
        self.fitness = float("inf")
        self.job_machine_dict = job_machine_dict
        self._validate_sequence()

    def _validate_sequence(self):
        """Ensure the sequence contains no None values and matches expected jobs."""
        if None in self.sequence:
            raise ValueError("Sequence contains None values")

        # Check all expected jobs are present
        job_counts = defaultdict(int)
        for op in self.sequence:
            if op is None:
                continue
            job_id, _ = op
            job_counts[job_id] += 1

        for job_id, ops in self.job_machine_dict.items():
            if job_counts[job_id] != len(ops):
                raise ValueError(f"Job {job_id} has incorrect number of operations")

    def is_valid(self) -> bool:
        """Check if the chromosome's sequence respects operation order constraints."""
        try:
            self._validate_sequence()

            for job_id in self.job_machine_dict:
                job_ops_in_sequence = [op[1] for op in self.sequence if op[0] == job_id]
                if job_ops_in_sequence != self.job_machine_dict[job_id]:
                    return False
            return True
        except (ValueError, TypeError, IndexError):
            return False


class GAOptimizer:
    """Genetic Algorithm optimizer for JSSP with robust operators."""

    def __init__(self, jssp):
        self.jssp = jssp
        self.iteration_history = []
        self.makespan_history = []
        self.diversity_history = []
        self.stagnation_count = 0

    def generate_initial_sequence(self) -> List[Tuple[int, int]]:
        """Generates a valid initial sequence preserving operation order within jobs."""
        remaining_ops = deepcopy(self.jssp.job_machine_dict)
        sequence = []

        while any(remaining_ops.values()):
            available_jobs = [j for j, ops in remaining_ops.items() if ops]
            job = random.choice(available_jobs)
            op_idx = remaining_ops[job].pop(0)
            sequence.append((job, op_idx))

        return sequence

    def initialize_population(self, population_size: int) -> List[Chromosome]:
        """Initialize the population with valid chromosomes."""
        population = []
        for _ in range(population_size):
            sequence = self.generate_initial_sequence()
            population.append(Chromosome(sequence, self.jssp.job_machine_dict))
        return population

    def calculate_diversity(self, population: List[Chromosome]) -> float:
        """Calculate population diversity based on position differences."""
        if not population:
            return 0.0

        centroid = [0] * len(population[0].sequence)
        for chromosome in population:
            for i, (job, _) in enumerate(chromosome.sequence):
                centroid[i] += job

        centroid = [x / len(population) for x in centroid]

        diversity = 0.0
        for chromosome in population:
            distance = sum(
                (p[0] - c) ** 2 for p, c in zip(chromosome.sequence, centroid)
            )
            diversity += math.sqrt(distance)

        return diversity / len(population)

    def tournament_selection(
        self, population: List[Chromosome], tournament_size: int
    ) -> Chromosome:
        """Select a chromosome using tournament selection."""
        tournament = random.sample(population, tournament_size)
        return min(tournament, key=attrgetter("fitness"))

    def order_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        """Robust OX implementation with fallback."""
        size = len(parent1.sequence)
        attempts = 0
        max_attempts = 10

        while attempts < max_attempts:
            attempts += 1
            child_sequence = [None] * size
            a, b = sorted(random.sample(range(size), 2))

            # Copy segment from parent1
            child_sequence[a:b] = parent1.sequence[a:b]

            # Fill from parent2
            ptr = 0
            for i in chain(range(a), range(b, size)):
                while ptr < size:
                    candidate = parent2.sequence[ptr]
                    ptr += 1

                    if candidate in child_sequence[a:b]:
                        continue

                    job_id, op_idx = candidate
                    existing_ops = [
                        op[1]
                        for op in child_sequence
                        if op is not None and op[0] == job_id
                    ]
                    expected_op = self.jssp.job_machine_dict[job_id][len(existing_ops)]

                    if op_idx == expected_op:
                        child_sequence[i] = candidate
                        break

            if None not in child_sequence:
                child = Chromosome(child_sequence, self.jssp.job_machine_dict)
                if child.is_valid():
                    return child

        return deepcopy(random.choice([parent1, parent2]))

    def swap_mutation(self, chromosome: Chromosome, mutation_rate: float) -> Chromosome:
        """Safe swap mutation that maintains constraints."""
        if random.random() >= mutation_rate:
            return deepcopy(chromosome)

        size = len(chromosome.sequence)
        if size < 2:
            return deepcopy(chromosome)

        attempts = 0
        max_attempts = 20

        while attempts < max_attempts:
            attempts += 1
            i, j = random.sample(range(size), 2)
            if chromosome.sequence[i][0] == chromosome.sequence[j][0]:
                continue

            mutated = deepcopy(chromosome)
            mutated.sequence[i], mutated.sequence[j] = (
                mutated.sequence[j],
                mutated.sequence[i],
            )

            if mutated.is_valid():
                return mutated

        return deepcopy(chromosome)

    def scramble_mutation(
        self, chromosome: Chromosome, mutation_rate: float
    ) -> Chromosome:
        """Safe scramble mutation that maintains constraints."""
        if random.random() >= mutation_rate:
            return deepcopy(chromosome)

        size = len(chromosome.sequence)
        if size < 2:
            return deepcopy(chromosome)

        attempts = 0
        max_attempts = 10

        while attempts < max_attempts:
            attempts += 1
            mutated = deepcopy(chromosome)
            a, b = sorted(random.sample(range(size), 2))

            # Group operations by job in the segment
            job_ops = defaultdict(list)
            for op in mutated.sequence[a:b]:
                job_ops[op[0]].append(op[1])

            # Scramble each job's operations
            scrambled = []
            for job_id in job_ops:
                ops = job_ops[job_id]
                random.shuffle(ops)
                scrambled.extend([(job_id, op) for op in ops])

            # Shuffle job order while maintaining operation sequences
            random.shuffle(scrambled)

            # Rebuild sequence
            mutated.sequence = mutated.sequence[:a] + scrambled + mutated.sequence[b:]

            if mutated.is_valid():
                return mutated

        return deepcopy(chromosome)

    def handle_stagnation(self, population: List[Chromosome], best_fitness: float):
        """Diversification strategies when stagnating."""
        population.sort(key=attrgetter("fitness"))
        num_to_replace = max(1, len(population) // 5)

        # Replace worst individuals
        for i in range(-num_to_replace, 0):
            population[i] = Chromosome(
                self.generate_initial_sequence(), self.jssp.job_machine_dict
            )

        # Apply additional mutation
        for i in range(len(population) // 2):
            if random.random() < 0.7:
                population[i] = self.scramble_mutation(population[i], 1.0)

    def optimize(
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
        """Run the robust GA optimization."""
        population = self.initialize_population(population_size)
        best_chromosome = None
        best_fitness = float("inf")
        self.stagnation_count = 0

        for iteration in range(max_iter):
            # Evaluation
            improved = False
            for chromosome in population:
                try:
                    chromosome.fitness = self.jssp.evaluate_schedule(
                        chromosome.sequence
                    )
                except:
                    chromosome.fitness = float("inf")

                if chromosome.fitness < best_fitness:
                    best_fitness = chromosome.fitness
                    best_chromosome = deepcopy(chromosome)
                    improved = True

            # Update stagnation counter
            self.stagnation_count = 0 if improved else self.stagnation_count + 1

            # Record metrics
            self.iteration_history.append(iteration)
            self.makespan_history.append(best_fitness)
            self.diversity_history.append(self.calculate_diversity(population))

            # Early stopping
            if (
                early_stopping_window
                and len(self.makespan_history) >= early_stopping_window
            ):
                window_min = min(self.makespan_history[-early_stopping_window:])
                if (best_fitness - window_min) < improvement_threshold * best_fitness:
                    print(f"Early stopping at iteration {iteration}")
                    break

            # Create new population
            new_population = []

            # Elitism: keep the best chromosome
            if best_chromosome:
                new_population.append(deepcopy(best_chromosome))

            # Generate offspring
            while len(new_population) < population_size:
                parent1 = self.tournament_selection(population, tournament_size)
                parent2 = self.tournament_selection(population, tournament_size)

                if random.random() < crossover_rate:
                    child = self.order_crossover(parent1, parent2)
                else:
                    child = deepcopy(random.choice([parent1, parent2]))

                # Apply mutations
                child = self.swap_mutation(child, mutation_rate)
                if random.random() < 0.3:
                    child = self.scramble_mutation(child, 1.0)

                # Ensure validity before adding
                if child.is_valid():
                    new_population.append(child)
                else:
                    new_population.append(deepcopy(parent1))

                if len(new_population) >= population_size:
                    break

            population = new_population[:population_size]

            # Stagnation handling
            if self.stagnation_count >= max_stagnation:
                self.handle_stagnation(population, best_fitness)
                self.stagnation_count = 0

            # Progress reporting
            if iteration % 10 == 0 or iteration == max_iter - 1:
                avg_fitness = sum(c.fitness for c in population) / len(population)
                print(
                    f"Iter {iteration}: Best={best_fitness:.1f} "
                    f"Avg={avg_fitness:.1f} "
                    f"Div={self.diversity_history[-1]:.2f} "
                    f"Stag={self.stagnation_count}/{max_stagnation}"
                )

        return best_chromosome.sequence, best_fitness
