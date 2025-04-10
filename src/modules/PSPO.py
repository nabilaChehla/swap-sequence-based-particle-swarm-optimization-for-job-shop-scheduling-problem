from copy import deepcopy
from typing import List, Tuple
import random
import numpy as np
import matplotlib.pyplot as plt
from time import time

W_END = 0.3
MAX_PRIORITY = 100.0


class PriorityParticle:
    """Particle for priority-based PSO that strictly preserves operation order within jobs."""

    def __init__(
        self, sequence: List[Tuple[int, int]], job_machine_dict: dict[int, list[int]]
    ):
        self.job_machine_dict = job_machine_dict
        self.sequence_length = len(sequence)

        # Create a list of all operations in order [(job1, op0), (job1, op1), ..., (jobN, opM)]
        self.all_operations = []
        for job in sorted(job_machine_dict.keys()):
            for op in range(len(job_machine_dict[job])):
                self.all_operations.append((job, op))

        self.position = self.sequence_to_priority(sequence)
        self.velocity = np.random.uniform(-1, 1, len(self.position))
        self.best_position = deepcopy(self.position)
        self.best_fitness = float("inf")
        self.fitness = float("inf")

    def sequence_to_priority(self, sequence: List[Tuple[int, int]]) -> np.ndarray:
        """Convert sequence to priority values while preserving operation order."""
        # Create a mapping from (job, op) to its index in the sequence
        sequence_positions = {(job, op): idx for idx, (job, op) in enumerate(sequence)}

        priorities = np.zeros(len(self.all_operations))

        # For each job, assign increasing priorities to operations
        for job in self.job_machine_dict:
            for op in range(len(self.job_machine_dict[job])):
                idx = self.all_operations.index((job, op))
                priorities[idx] = sequence_positions[(job, op)]

        # Normalize priorities to be between 0 and MAX_PRIORITY
        if len(priorities) > 0:
            priorities = (
                (priorities - priorities.min())
                / (priorities.max() - priorities.min() + 1e-10)
                * MAX_PRIORITY
            )

        return priorities

    def priority_to_sequence(self, priority: np.ndarray) -> List[Tuple[int, int]]:
        """Convert priorities to valid sequence while strictly preserving operation order."""
        # Group operations by job and sort by priority within each job
        job_operations = {}
        for idx, (job, op) in enumerate(self.all_operations):
            if job not in job_operations:
                job_operations[job] = []
            job_operations[job].append((op, priority[idx]))

        # Sort operations within each job by their priority
        for job in job_operations:
            job_operations[job].sort(key=lambda x: x[1])

        # Create a sequence by selecting the next operation from each job
        sequence = []
        op_pointers = {job: 0 for job in job_operations}
        remaining_ops = sum(len(ops) for ops in job_operations.values())

        while remaining_ops > 0:
            # Find all jobs that have operations left to schedule
            available_jobs = [
                job
                for job in job_operations
                if op_pointers[job] < len(job_operations[job])
            ]

            # Select the job with the highest priority next operation
            selected_job = min(
                available_jobs, key=lambda j: job_operations[j][op_pointers[j]][1]
            )

            # Add the next operation from this job to the sequence
            op = job_operations[selected_job][op_pointers[selected_job]][0]
            sequence.append((selected_job, op))
            op_pointers[selected_job] += 1
            remaining_ops -= 1

        return sequence


class PriorityParticle:
    def __init__(
        self, sequence: List[Tuple[int, int]], job_machine_dict: dict[int, list[int]]
    ):
        self.job_machine_dict = job_machine_dict
        self.sequence_length = len(sequence)

        # Create a list of all operations in order [(job1, op0), (job1, op1), ..., (jobN, opM)]
        self.all_operations = []
        for job in sorted(job_machine_dict.keys()):
            for op in range(len(job_machine_dict[job])):
                self.all_operations.append((job, op))

        # Initialize with random priorities between 0 and MAX_PRIORITY
        self.position = np.random.uniform(0, MAX_PRIORITY, len(self.all_operations))

        # Ensure operation order constraints by modifying the random priorities
        for job in job_machine_dict:
            for op in range(1, len(job_machine_dict[job])):
                prev_idx = self.all_operations.index((job, op - 1))
                curr_idx = self.all_operations.index((job, op))
                # Ensure current operation has higher priority than previous
                if self.position[curr_idx] <= self.position[prev_idx]:
                    self.position[curr_idx] = self.position[prev_idx] + random.uniform(
                        0.1, 1.0
                    )

        self.velocity = np.random.uniform(-1, 1, len(self.position))
        self.best_position = deepcopy(self.position)
        self.best_fitness = float("inf")
        self.fitness = float("inf")

    def priority_to_sequence(self, priority: np.ndarray) -> List[Tuple[int, int]]:
        """Convert priorities to valid sequence while strictly preserving operation order."""
        # Create a list of (job, op, priority) tuples
        ops_with_priority = []
        for idx, (job, op) in enumerate(self.all_operations):
            ops_with_priority.append((job, op, priority[idx]))

        # Sort by priority while maintaining operation order constraints
        sorted_ops = []
        remaining_ops = ops_with_priority.copy()
        job_pointers = {job: 0 for job in self.job_machine_dict}

        while remaining_ops:
            # Find all operations that are the next in their job sequence
            candidates = []
            for job in self.job_machine_dict:
                if job_pointers[job] < len(self.job_machine_dict[job]):
                    op_idx = job_pointers[job]
                    candidates.append((job, op_idx))

            # From these candidates, select the one with highest priority
            if candidates:
                # Find the actual operation in remaining_ops
                available = []
                for job, op in candidates:
                    for item in remaining_ops:
                        if item[0] == job and item[1] == op:
                            available.append(item)
                            break

                if available:
                    selected = min(available, key=lambda x: x[2])
                    sorted_ops.append((selected[0], selected[1]))
                    remaining_ops.remove(selected)
                    job_pointers[selected[0]] += 1

        return sorted_ops

    def update_velocity(
        self, global_best_position: np.ndarray, w: float, c1: float, c2: float
    ):
        """Update velocity with standard PSO formula (no constriction)"""
        r1 = random.random()
        r2 = random.random()

        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)

        self.velocity = w * self.velocity + cognitive + social

        # Velocity clamping
        max_velocity = MAX_PRIORITY * 0.1
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)

    def update_position(self):
        """Update position with velocity, then enforce operation order constraints"""
        self.position = np.clip(self.position + self.velocity, 0, MAX_PRIORITY)

        # Ensure operation order constraints are maintained
        for job in self.job_machine_dict:
            for op in range(1, len(self.job_machine_dict[job])):
                prev_idx = self.all_operations.index((job, op - 1))
                curr_idx = self.all_operations.index((job, op))
                if self.position[curr_idx] <= self.position[prev_idx]:
                    self.position[curr_idx] = self.position[prev_idx] + 0.1

    def generate_initial_sequence(self, cluster_size: int = 3) -> List[Tuple[int, int]]:
        """Cluster approach that balances machine workload during clustering."""
        remaining_ops = deepcopy(self.jssp.job_machine_dict)
        sequence = []
        machine_counts = {m: 0 for m in range(1, self.jssp.num_machines + 1)}

        while any(ops_left for ops_left in remaining_ops.values()):
            available_ops = []
            for job_idx, ops_left in remaining_ops.items():
                if ops_left:
                    op_idx = ops_left[0]
                    op = self.jssp.jobs[job_idx].operations[op_idx]
                    available_ops.append(
                        (job_idx, op_idx, op.processing_time, op.machine)
                    )

            # Balance clusters by machine distribution
            clusters = []
            current_cluster = []
            machine_in_cluster = set()

            for op in sorted(available_ops, key=lambda x: machine_counts[x[3]]):
                if (
                    len(current_cluster) < cluster_size
                    and op[3] not in machine_in_cluster
                ):
                    current_cluster.append(op)
                    machine_in_cluster.add(op[3])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [op]
                    machine_in_cluster = {op[3]}
            if current_cluster:
                clusters.append(current_cluster)

            # Process clusters
            for cluster in clusters:
                # Sort by both processing time and machine load
                cluster_sorted = sorted(
                    cluster, key=lambda x: (machine_counts[x[3]], x[2])
                )

                for op in cluster_sorted:
                    job_idx, op_idx, _, machine = op
                    if remaining_ops[job_idx] and op_idx == remaining_ops[job_idx][0]:
                        sequence.append((job_idx, op_idx))
                        remaining_ops[job_idx].pop(0)
                        machine_counts[machine] += 1

        return sequence

    def calculate_diversity(self, particles: List[PriorityParticle]) -> float:
        """Calculate population diversity with NaN protection."""
        if not particles or any(np.isnan(p.position).any() for p in particles):
            return 0.0
        try:
            centroid = np.nanmean([p.position for p in particles], axis=0)
            distances = [
                np.linalg.norm(p.position - centroid)
                for p in particles
                if not np.isnan(p.position).any()
            ]
            return np.nanmean(distances) if distances else 0.0
        except:
            return 0.0

    def local_search(self, sequence, max_tries=100):
        best_seq = sequence
        best_fitness = self.jssp.evaluate_schedule(sequence)

        for _ in range(max_tries):
            # Find critical path and focus swaps there
            i = random.randint(0, len(sequence) - 2)
            if sequence[i][0] == sequence[i + 1][0]:
                continue  # Skip operations from same job

            new_seq = sequence[:i] + [sequence[i + 1], sequence[i]] + sequence[i + 2 :]
            new_fitness = self.jssp.evaluate_schedule(new_seq)

            if new_fitness < best_fitness:
                best_seq, best_fitness = new_seq, new_fitness

        return best_seq, best_fitness

    def apply_mutation(self, particle: PriorityParticle, mutation_rate: float = 0.1):
        """Apply mutation to a particle's position to escape local optima."""
        # Randomly select some dimensions to mutate
        mutation_mask = np.random.rand(len(particle.position)) < mutation_rate
        mutation_values = np.random.uniform(
            -MAX_PRIORITY / 4, MAX_PRIORITY / 4, len(particle.position)
        )

        # Apply mutation while keeping values within bounds
        particle.position = np.clip(
            particle.position + mutation_mask * mutation_values, 0, MAX_PRIORITY
        )

        # Update velocity to reflect the mutation
        particle.velocity = np.clip(
            particle.velocity + mutation_mask * mutation_values, -1.0, 1.0
        )

    def diversify_population(
        self,
        particles: List[PriorityParticle],
        best_particles_to_keep: int = 5,
        mutation_rate: float = 0.2,
    ):
        """Diversify the population while keeping elite solutions."""
        # Sort particles by fitness
        particles.sort(key=lambda p: p.best_fitness)

        # Keep the best particles unchanged
        elite_particles = particles[:best_particles_to_keep]

        # Apply mutation to some of the remaining particles
        for p in particles[best_particles_to_keep:-best_particles_to_keep]:
            if random.random() < 0.7:  # 70% chance to mutate
                self.apply_mutation(p, mutation_rate)

        # Replace the worst particles with new random particles
        for i in range(len(particles) - best_particles_to_keep, len(particles)):
            sequence = self.generate_initial_sequence()
            particles[i] = PriorityParticle(sequence, self.jssp.job_machine_dict)

    def optimize(
        self,
        num_particles: int = 50,
        max_iter: int = 500,
        w: float = 0.7,
        c1: float = 1.7,
        c2: float = 1.7,
        mutation_rate: bool = 0.3,
        adaptive_params: bool = True,
        max_stagnation: int = 30,
        early_stopping_window: int = 50,
        improvement_threshold: float = 0.005,
        apply_local_search: bool = True,
        verbose: bool = True,
    ) -> Tuple[List[Tuple[int, int]], float, float]:
        """Run the PSO optimization while preserving operation order."""
        particles = []
        global_best_position = None
        global_best_fitness = float("inf")
        global_best_sequence = None
        self.stagnation_count = 0
        self.start_time = time()

        # Initialize particles with valid sequences
        for _ in range(num_particles):
            sequence = self.generate_initial_sequence()
            particle = PriorityParticle(sequence, self.jssp.job_machine_dict)
            particles.append(particle)

        for iteration in range(max_iter):
            current_w = w
            if adaptive_params:
                current_w = max(W_END, w - (w - W_END) * (iteration / max_iter))

            improved = False
            for particle in particles:
                sequence = particle.priority_to_sequence(particle.position)

                # Apply local search if enabled
                if apply_local_search and random.random() < 0.3:
                    sequence, fitness = self.local_search(sequence)
                else:
                    fitness = self.jssp.evaluate_schedule(sequence)

                particle.fitness = fitness

                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = deepcopy(particle.position)

                if particle.fitness < global_best_fitness:
                    global_best_fitness = particle.fitness
                    global_best_position = deepcopy(particle.position)
                    global_best_sequence = deepcopy(sequence)
                    improved = True

            self.stagnation_count = 0 if improved else self.stagnation_count + 1
            self.iteration_history.append(iteration)
            self.makespan_history.append(global_best_fitness)
            self.diversity_history.append(self.calculate_diversity(particles))

            # Apply mutation to random particles (even when not stagnating)
            if (
                random.random() < mutation_rate
            ):  # 20% chance to mutate a random particle
                self.apply_mutation(
                    random.choice(particles),
                    mutation_rate=0.1 + 0.1 * (self.stagnation_count / max_stagnation),
                )

            # Enhanced stagnation handling
            if self.stagnation_count >= max_stagnation // 2:
                # Increase mutation rate when approaching stagnation
                for p in random.sample(particles, max(1, int(0.3 * len(particles)))):
                    self.apply_mutation(p, mutation_rate)

            if self.stagnation_count >= max_stagnation:
                # More aggressive diversification when fully stagnated
                self.diversify_population(particles, best_particles_to_keep=5)
                self.stagnation_count = 0  # Reset after diversification

            for particle in particles:
                particle.update_velocity(global_best_position, current_w, c1, c2)
                particle.update_position()

            if (
                early_stopping_window
                and len(self.makespan_history) >= early_stopping_window
            ):
                window_min = min(self.makespan_history[-early_stopping_window:])
                if (
                    global_best_fitness - window_min
                ) < improvement_threshold * global_best_fitness:
                    if verbose:
                        print(f"Early stopping at iteration {iteration}")
                    break

            if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
                elapsed = time() - self.start_time
                print(
                    f"Iter {iteration}: Best={global_best_fitness:.1f} "
                    f"Div={self.diversity_history[-1]:.2f} "
                    f"Stag={self.stagnation_count}/{max_stagnation} "
                    f"Time={elapsed:.1f}s"
                )

        exec_time = time() - self.start_time
        return global_best_sequence, global_best_fitness, exec_time

    def plot_convergence(self):
        """Plot the convergence history."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.iteration_history, self.makespan_history)
        plt.title("Makespan Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Makespan")

        plt.subplot(1, 2, 2)
        plt.plot(self.iteration_history, self.diversity_history)
        plt.title("Population Diversity")
        plt.xlabel("Iteration")
        plt.ylabel("Diversity")

        plt.tight_layout()
        plt.show()
