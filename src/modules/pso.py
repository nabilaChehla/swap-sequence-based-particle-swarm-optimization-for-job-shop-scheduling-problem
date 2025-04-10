import random
from copy import deepcopy
from typing import List, Tuple
from src.modules.modelisation import JSSP
from src.modules.particle import Particle
import math
from collections import defaultdict

# Stagnation adaptation constants (add near top of file with other constants)
MIN_W = 0.2
MAX_MUTATION = 1
MAX_ATTEMPTS = 50

# Stagnation adaptation parameters
COGNITIVE_BOOST_FACTOR = 0.5  # How much to increase c1 during stagnation
SOCIAL_REDUCTION_FACTOR = 0.3  # How much to decrease c2 during stagnation
MIN_SOCIAL_FACTOR = 0.5  # Minimum allowed value for c2 during adaptation
MAX_COGNITIVE_BOOST = 1.5  # 1 + COGNITIVE_BOOST_FACTOR
MAX_SOCIAL_REDUCTION = 0.7  # 1 - SOCIAL_REDUCTION_FACTOR


class PSOOptimizer:
    """Enhanced PSO optimization process with advanced stagnation handling."""

    def __init__(self, jssp: JSSP):
        self.jssp = jssp
        self.iteration_history = []
        self.makespan_history = []
        self.diversity_history = []
        self.stagnation_count = 0
        self.intensity_stagnation = 0
        self.last_improvement = 0

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

    def calculate_diversity(self, particles: List[Particle]) -> float:
        """Calculate population diversity based on position differences."""
        if not particles:
            return 0.0

        centroid = [0] * len(particles[0].position)
        for particle in particles:
            for i, (job, machine) in enumerate(particle.position):
                centroid[i] += job

        centroid = [x / len(particles) for x in centroid]

        diversity = 0.0
        for particle in particles:
            distance = sum((p[0] - c) ** 2 for p, c in zip(particle.position, centroid))
            diversity += math.sqrt(distance)

        return diversity / len(particles)

    def is_sequence_valid(self, sequence: List[Tuple[int, int]], job_id: int) -> bool:
        """Check if operations for a job are in correct machine order."""
        job_ops = [op[1] for op in sequence if op[0] == job_id]
        return job_ops == self.jssp.job_machine_dict[job_id]

    def generate_random_sequence(self) -> List[Tuple[int, int]]:
        """Generate a completely random valid sequence."""
        sequence = []
        remaining_ops = deepcopy(self.jssp.job_machine_dict)

        while any(ops_left for ops_left in remaining_ops.values()):
            available_ops = []
            for job_idx, ops_left in remaining_ops.items():
                if ops_left:
                    available_ops.append((job_idx, ops_left[0]))

            selected = random.choice(available_ops)
            sequence.append(selected)
            remaining_ops[selected[0]].pop(0)

        return sequence

    def crossover(self, parent1: Particle, parent2: Particle) -> Particle:
        """Order-based crossover for scheduling problems."""
        child_seq = []
        p1_seq = parent1.best_position
        p2_seq = parent2.best_position

        point = random.randint(1, len(p1_seq) - 2)
        child_seq = deepcopy(p1_seq[:point])

        remaining_ops = defaultdict(int)
        for job_idx, op_idx in p2_seq:
            remaining_ops[job_idx] += 1

        for job_idx, op_idx in child_seq:
            if (
                remaining_ops[job_idx] > 0
                and op_idx
                == self.jssp.job_machine_dict[job_idx][-remaining_ops[job_idx]]
            ):
                remaining_ops[job_idx] -= 1

        for job_idx, op_idx in p2_seq:
            if remaining_ops[job_idx] > 0:
                next_op = self.jssp.job_machine_dict[job_idx][-remaining_ops[job_idx]]
                child_seq.append((job_idx, next_op))
                remaining_ops[job_idx] -= 1

        return Particle(child_seq, self.jssp.job_machine_dict)

    def path_relinking(
        self, solution1: List[Tuple[int, int]], solution2: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Generate intermediate solutions between two good solutions."""
        diff_positions = [
            i for i in range(len(solution1)) if solution1[i] != solution2[i]
        ]

        if not diff_positions:
            return deepcopy(solution1)

        num_changes = max(1, len(diff_positions) // 3)
        positions_to_change = random.sample(diff_positions, num_changes)

        new_solution = deepcopy(solution1)
        for pos in positions_to_change:
            op_to_insert = solution2[pos]
            if op_to_insert in new_solution:
                new_solution.remove(op_to_insert)
            new_solution.insert(pos, op_to_insert)

            if not self.is_sequence_valid(new_solution, op_to_insert[0]):
                new_solution = deepcopy(solution1)
                continue

        return new_solution

    def perturb_solution(
        self, solution: List[Tuple[int, int]], num_swaps: int = 3
    ) -> List[Tuple[int, int]]:
        """Create a slightly modified version of the best solution."""
        perturbed = deepcopy(solution)
        swaps_applied = 0
        attempts = 0

        while swaps_applied < num_swaps and attempts < MAX_ATTEMPTS:
            i, j = random.sample(range(len(perturbed)), 2)
            if perturbed[i][0] != perturbed[j][0]:
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

                if self.is_sequence_valid(
                    perturbed, perturbed[i][0]
                ) and self.is_sequence_valid(perturbed, perturbed[j][0]):
                    swaps_applied += 1
                else:
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            attempts += 1

        return perturbed

    def handle_stagnation(
        self, particles: List[Particle], global_best_position: List[Tuple[int, int]]
    ):
        """Multi-level diversification strategies based on stagnation intensity."""
        stagnation_level = min(3, 1 + self.stagnation_count // 5)

        if stagnation_level == 1:
            particles.sort(key=lambda p: p.best_fitness)
            num_to_reinit = max(1, len(particles) // 5)

            for i in range(-num_to_reinit, 0):
                particles[i] = Particle(
                    self.generate_initial_sequence(), self.jssp.job_machine_dict
                )

            perturbed = self.perturb_solution(
                global_best_position, max(len(global_best_position) // 10, 1)
            )
            particles[-1] = Particle(perturbed, self.jssp.job_machine_dict)

        elif stagnation_level == 2:
            particles.sort(key=lambda p: p.best_fitness)
            for i in range(len(particles) // 2, len(particles)):
                if random.random() < 0.7:
                    particles[i] = Particle(
                        self.generate_initial_sequence(), self.jssp.job_machine_dict
                    )

            perturbed = self.perturb_solution(
                global_best_position, max(len(global_best_position) // 3, 2)
            )
            particles[random.randint(0, len(particles) - 1)] = Particle(
                perturbed, self.jssp.job_machine_dict
            )

            particles[random.randint(0, len(particles) - 1)] = Particle(
                self.generate_random_sequence(), self.jssp.job_machine_dict
            )

        else:
            particles.sort(key=lambda p: p.best_fitness)
            num_to_keep = max(2, len(particles) // 10)

            for i in range(num_to_keep, len(particles)):
                if random.random() < 0.5:
                    particles[i] = Particle(
                        self.generate_initial_sequence(), self.jssp.job_machine_dict
                    )
                else:
                    parent1 = random.choice(particles[:num_to_keep])
                    parent2 = Particle(
                        self.generate_initial_sequence(), self.jssp.job_machine_dict
                    )
                    particles[i] = self.crossover(parent1, parent2)

            if num_to_keep >= 2:
                new_solution = self.path_relinking(
                    particles[0].best_position, particles[1].best_position
                )
                particles[-1] = Particle(new_solution, self.jssp.job_machine_dict)

    def optimize(
        self,
        num_particles: int = 30,
        max_iter: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        adaptive_params: bool = True,
        mutation_rate: float = 1,
        max_stagnation: int = 15,
        early_stopping_window: int = 20,
        improvement_threshold: float = 0.01,
    ):
        """Run the enhanced PSO optimization with adaptive stagnation handling."""
        particles = []
        global_best_position = None
        global_best_fitness = float("inf")
        self.stagnation_count = 0

        # Initialize swarm
        for _ in range(num_particles):
            sequence = self.generate_initial_sequence()
            particles.append(Particle(sequence, self.jssp.job_machine_dict))

        for iteration in range(max_iter):
            # Adaptive parameters
            current_w = w
            current_c1 = c1
            current_c2 = c2
            current_mutation = mutation_rate

            if adaptive_params:
                stagnation_ratio = self.stagnation_count / max_stagnation

                # Cognitive component adaptation
                current_c1 = c1 * (1 + COGNITIVE_BOOST_FACTOR * stagnation_ratio)

                # Social component adaptation
                current_c2 = c2 * max(
                    MIN_SOCIAL_FACTOR,
                    1 - SOCIAL_REDUCTION_FACTOR * stagnation_ratio,
                )

                # Inertia weight adaptation
                current_w = max(MIN_W, w * (1 - stagnation_ratio))

                # Mutation rate adaptation
                current_mutation = min(
                    MAX_MUTATION, mutation_rate * (1 + 2 * stagnation_ratio)
                )

            improved = False
            for particle in particles:
                particle.fitness = self.jssp.evaluate_schedule(particle.position)

                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = deepcopy(particle.position)

                if particle.fitness < global_best_fitness:
                    global_best_fitness = particle.fitness
                    global_best_position = deepcopy(particle.position)
                    improved = True

            diversity = self.calculate_diversity(particles)
            if not improved:
                if diversity < 0.1 * (
                    self.diversity_history[0] if self.diversity_history else 1.0
                ):
                    self.stagnation_count += 2
                else:
                    self.stagnation_count += 1

            self.iteration_history.append(iteration)
            self.makespan_history.append(global_best_fitness)
            self.diversity_history.append(diversity)

            for particle in particles:
                particle.update_velocity(
                    global_best_position,
                    current_w,
                    current_c1,
                    current_c2,
                    current_mutation,
                )
                particle.update_position()
                particle.apply_mutation(current_mutation)

            effective_stagnation_threshold = max(
                5, max_stagnation * (1 - 0.5 * (iteration / max_iter))
            )

            if self.stagnation_count >= effective_stagnation_threshold:
                self.handle_stagnation(particles, global_best_position)
                self.stagnation_count = 0
                current_w = max(MIN_W, w * 0.8)
                current_c1 = c1 * 1.2
                current_c2 = c2 * 0.8
                current_mutation = min(MAX_MUTATION, mutation_rate * 1.5)

            if (
                early_stopping_window
                and len(self.makespan_history) >= early_stopping_window
                and (
                    global_best_fitness
                    - min(self.makespan_history[-early_stopping_window:])
                )
                < improvement_threshold * global_best_fitness
            ):
                print(f"Early stopping at iteration {iteration}")
                break

            if iteration % 10 == 0 or iteration == max_iter - 1:
                print(
                    f"Iter {iteration}: Best={global_best_fitness:.1f} "
                    f"Div={self.diversity_history[-1]:.2f} "
                    f"Stag={self.stagnation_count}/{max_stagnation}"
                )

        return global_best_position, global_best_fitness
