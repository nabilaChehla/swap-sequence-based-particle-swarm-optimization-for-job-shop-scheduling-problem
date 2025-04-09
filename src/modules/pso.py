import random
from copy import deepcopy
from typing import List, Tuple
from src.modules.modelisation import JSSP
from src.modules.particle import Particle
import math

DIVERSITY_THRESHOLD = 0.1  # Threshold for diversity
MIN_W = 0.2 # minimial number of iterations
MAX_MUTATION = 1 # Maximum mutation rate
MAX_ATTEMPTS = 100 # Maximum attempts for mutation
STAGNATION = 4 # Number of particles to reinitialize when stagnating

class PSOOptimizer:
    """Enhanced PSO optimization process with stagnation handling."""

    def __init__(self, jssp: JSSP):
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

    def calculate_diversity(self, particles: List[Particle]) -> float:
        """Calculate population diversity based on position differences.
        L’idée ici est de mesurer à quel point les positions des particules sont diverses 
        par rapport à un centroïde de la population. Plus la diversité est grande, plus la population 
        explore différents points de l'espace de recherche.
        
        """
        if not particles:
            return 0.0

        centroid = [0] * len(particles[0].position) # Initialisation du centroid 

        # Calcul du centroïde
        for particle in particles:
            for i, (job, machine) in enumerate(particle.position):
                centroid[i] += job # chaque élément du centroid est un total des ID des jobs pour toutes les particules.

        centroid = [x / len(particles) for x in centroid] # Moyenne des IDs de jobs.

        diversity = 0.0
        for particle in particles:
            distance = sum((p[0] - c) ** 2 for p, c in zip(particle.position, centroid)) # mesures la distance euclidienne entre chaque particule et le centroid
            diversity += math.sqrt(distance)

        # diversité moyenne 
        return diversity / len(particles)

    # def calculate_diversity(self, particles: List[Particle]) -> float:
    #     """Calcule la diversité de la population en utilisant la distance de Hamming.
        
    #     La distance de Hamming compte combien d'opérations sont à des positions différentes
    #     entre deux particules. Plus la diversité moyenne est grande, plus la population est dispersée.
    #     """
    #     if not particles or len(particles) < 2:
    #         return 0.0

    #     total_distance = 0
    #     count = 0

    #     # Comparer chaque paire unique de particules
    #     for i in range(len(particles)):
    #         for j in range(i + 1, len(particles)):
    #             pos1 = particles[i].position
    #             pos2 = particles[j].position

    #             # Hamming distance = nombre de différences entre les deux séquences
    #             hamming = sum(op1 != op2 for op1, op2 in zip(pos1, pos2))
    #             total_distance += hamming
    #             count += 1

    #     # Diversité moyenne
    #     return total_distance / count


    def is_sequence_valid(self, sequence: List[Tuple[int, int]], job_id: int) -> bool:
        """Check if operations for a job are in correct machine order."""
        job_ops = [op[1] for op in sequence if op[0] == job_id]
        return job_ops == self.jssp.job_machine_dict[job_id]

    def handle_stagnation(
        self, particles: List[Particle], global_best_position: List[Tuple[int, int]]
    ):
        """Diversification strategies when stagnating."""
        # Reinitialize worst particles
        particles.sort(key=lambda p: p.best_fitness)
        num_to_reinit = max(1, len(particles) // STAGNATION) # On choisit un nombre de particules à réinitialiser basé sur le taux de stagnation.

        for i in range(-num_to_reinit, 0):
            particles[i] = Particle(
                self.generate_initial_sequence(), self.jssp.job_machine_dict # Ces nouvelles particules sont ainsi réintroduites pour favoriser l'exploration de nouvelles zones de l'espace de recherche.
            )

        # Add perturbed global best : modifie légèrement la position de la meilleure solution pour l'empêcher de stagner.
        perturbed = self.perturb_solution( 
            global_best_position, max(len(global_best_position) // 5, 1)
        )
        particles[-1] = Particle(perturbed, self.jssp.job_machine_dict)

    def perturb_solution(
        self, solution: List[Tuple[int, int]], num_swaps: int = 3
    ) -> List[Tuple[int, int]]:
        """Create a slightly modified version of the best solution."""
        perturbed = deepcopy(solution)
        swaps_applied = 0
        attempts = 0

        while swaps_applied < num_swaps and attempts < MAX_ATTEMPTS:
            i, j = random.sample(range(len(perturbed)), 2) # choisit deux indices distincts i et j dans la solution perturbée. Cela permet de sélectionner deux positions de jobs à échanger.
            if perturbed[i][0] != perturbed[j][0]: # Assure que les jobs sont différents
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

                if self.is_sequence_valid(
                    perturbed, perturbed[i][0] # Vérifie si la séquence perturbée est valide après l'échange pour l'id du job de la position i.
                ) and self.is_sequence_valid(perturbed, perturbed[j][0]):
                    swaps_applied += 1
                else:
                    # Undo invalid swap
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            attempts += 1

        return perturbed

    def optimize(
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
        """Run the enhanced PSO optimization."""
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
            current_mutation = mutation_rate
            if adaptive_params:
                current_w = w * (1 - iteration / max_iter) + MIN_W * ( # Inertia w : It starts high and decreases as the iteration progresses (to encourage exploration early and exploitation later).
                    iteration / max_iter
                )
                current_mutation = min( # If there’s more stagnation, the mutation rate increases, allowing more exploration.
                    MAX_MUTATION, mutation_rate * (1 + self.stagnation_count / 10)  # nsures that the mutation rate doesn’t exceed MAX_MUTATION, so the algorithm doesn't go too random.
                ) 

            # Evaluate particles
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

            # Update stagnation counter
            self.stagnation_count = 0 if improved else self.stagnation_count + 1

            # Record metrics
            self.iteration_history.append(iteration)
            self.makespan_history.append(global_best_fitness)
            self.diversity_history.append(self.calculate_diversity(particles))

            # Update particles
            for particle in particles:
                particle.update_velocity(
                    global_best_position, current_w, c1, c2, mutation_rate # static mutation rate
                )
                particle.update_position()
                particle.apply_mutation(current_mutation) # dynamic changing mutation

            # Stagnation handling
            if self.stagnation_count >= max_stagnation: # If the stagnation count exceeds max_stagnation, meaning the swarm has not improved for a while
                self.handle_stagnation(particles, global_best_position)
                self.stagnation_count = 0

            # diversity = self.calculate_diversity(particles)
            # self.diversity_history.append(diversity)

            # if diversity < DIVERSITY_THRESHOLD * len(particles[0].position): # If the diversity is below a certain threshold, indicating that the particles are converging too closely together.
            #     print(f"Low diversity detected at iteration {iteration}, triggering diversification...")
            #     self.handle_stagnation(particles, global_best_position)
            #     self.stagnation_count = 0



            # Early stopping : to terminate the algorithm if no significant improvement in fitness occurs over the last 
            # early_stopping_window iterations. If the improvement is smaller than a certain threshold (improvement_threshold), the optimization stops early.
            if (
                early_stopping_window
                and len(self.makespan_history) >= early_stopping_window # On attend d’avoir accumulé au moins early_stopping_window itérations pour faire cette vérification.
            ):
                window_min = min(self.makespan_history[-early_stopping_window:]) # On regarde la meilleure valeur de fitness dans la fenêtre d’arrêt précoce.
                # On compare la différence entre la meilleure valeur de fitness globale et la meilleure valeur de fitness dans la fenêtre d’arrêt précoce.
                # Si cette différence est inférieure à un certain seuil (improvement_threshold), on arrête l’optimisation.
                if (
                    global_best_fitness - window_min
                ) < improvement_threshold * global_best_fitness:
                    print(f"Early stopping at iteration {iteration}")
                    break

            # Progress reporting
            if iteration % 10 == 0 or iteration == max_iter - 1:
                print(
                    f"Iter {iteration}: Best={global_best_fitness:.1f} "
                    f"Div={self.diversity_history[-1]:.2f} "
                    f"Stag={self.stagnation_count}/{max_stagnation}"
                )

        return global_best_position, global_best_fitness
