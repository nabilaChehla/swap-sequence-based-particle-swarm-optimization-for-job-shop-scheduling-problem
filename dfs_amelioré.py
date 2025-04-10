import numpy as np  
from copy import deepcopy  
import time 
from modelisation import Operation, Job, JSSP  

class JSSPSolverDFS:
    def __init__(self, jssp_instance):
        """Initialise le solveur avec une instance du problème JSSP"""
        self.jssp = jssp_instance  
        self.best_schedule = None  
        self.best_makespan = float('inf')  
        self.nodes_explored = 0  

    def solve(self):
        """Méthode principale pour lancer la résolution"""
        self.jssp.initialize_schedule()  # Réinitialise le planning
        self.DFS(self.jssp.jobs, [], 0)  # Lance la recherche DFS
        return self.best_schedule, self.best_makespan  # Retourne la meilleure solution

    def DFS(self, remaining_jobs, current_schedule, current_makespan):
        
        self.nodes_explored += 1  

        # Condition d'arrêt : plus de jobs à ordonnancer
        if not remaining_jobs:
            # Si la solution courante est meilleure, on la sauvegarde
            if current_makespan < self.best_makespan:
                self.best_makespan = current_makespan
                self.best_schedule = deepcopy(current_schedule)
            return

        # Tri des jobs par ordre croissant de temps restant 
        remaining_jobs.sort(
            key=lambda j: (
                # Critère principal : temps total restant pour le job
                sum(op.processing_time for op in j.operations[j.current_operation_index:]),
                # Critère secondaire : priorité aux jobs moins avancés (index bas)
                -j.current_operation_index 
            ),
            reverse=True  # Tri décroissant
        )

        # Exploration de chaque job restant
        for job in remaining_jobs:
            # Si le job a terminé toutes ses opérations, on passe au suivant
            if job.current_operation_index >= len(job.operations):
                continue

            # Récupération de l'opération courante à planifier
            op = job.operations[job.current_operation_index]
            machine_mapping = op.machine + 1  # Conversion 0-indexé → 1-indexé car dans modélisation on commence par zero 
            processing_time = op.processing_time  # Durée de l'opération

            # Calcul du temps de début possible :
            # 1. Temps de fin de l'opération précédente dans le même job
            last_op_end = 0 if job.current_operation_index == 0 else \
                job.operations[job.current_operation_index - 1].end_time

            # 2. Dernier temps d'utilisation de la machine concernée
            last_machine_time = max([op.end_time for op in self.jssp.schedule.get(machine_mapping, [])], default=0)

            # Le temps de début est le maximum entre ces deux contraintes
            start_time = max(last_op_end, last_machine_time)
            end_time = start_time + processing_time  # Temps de fin

            # Élagage : si cette branche ne peut pas améliorer la solution
            if end_time >= self.best_makespan:
                continue  # On abandonne cette branche

            # Mise à jour des attributs de l'opération
            op.start_time = start_time
            op.end_time = end_time
            # Ajout au planning central (1-indexé)
            self.jssp.schedule[machine_mapping].append(op)

            # Passage à l'opération suivante pour ce job
            job.current_operation_index += 1
            # Filtrage des jobs ayant encore des opérations à planifier
            new_remaining_jobs = [j for j in remaining_jobs if j.current_operation_index < len(j.operations)]

            # Appel récursif pour explorer cette branche
            self.DFS(
                new_remaining_jobs,  # Jobs restants
                current_schedule + [{  # Nouvelle entrée dans le planning
                    'job_id': job.job_id,
                    'operation': job.current_operation_index - 1,
                    'machine': op.machine,  # 0-indexé pour la sortie
                    'start_time': start_time,
                    'end_time': end_time
                }],
                max(current_makespan, end_time)  # Nouveau makespan courant
            )

            
            job.current_operation_index -= 1  # Revenir à l'opération précédente
            op.start_time = None  # Réinitialiser
            op.end_time = None  # Réinitialiser
            self.jssp.schedule[machine_mapping].remove(op)  # Retirer du planning

if __name__ == "__main__":
   
    machines_matrix = [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0],
        [2, 3, 4, 0, 1],
        [3, 4, 0, 1, 2],
        [4, 0, 1, 2, 3]
    ]

    times_matrix = [
        [5, 3, 2, 4, 1],
        [3, 4, 2, 1, 5],
        [2, 5, 3, 4, 1],
        [4, 1, 5, 2, 3],
        [1, 3, 4, 5, 2]
    ]

  
    jssp = JSSP(machines_matrix, times_matrix)  
    solver = JSSPSolverDFS(jssp)  

    
    start_time = time.time()
    best_schedule, best_makespan = solver.solve()
    execution_time = time.time() - start_time

    # Affichage des résultats
    print(f"Optimal Makespan: {best_makespan} (Expected: 4)")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Nodes Explored: {solver.nodes_explored}")

    # Affichage détaillé par machine
    print("\nMachine Schedules (0-indexed):")
    for m in range(len(machines_matrix[0])):  # Pour chaque machine
        # Filtrage et tri des opérations par machine
        machine_ops = [op for op in best_schedule if op['machine'] == m]
        machine_ops.sort(key=lambda x: x['start_time'])
        print(f"Machine {m}:")
        for op in machine_ops:
            print(f"  Job {op['job_id']}, Op {op['operation']}: {op['start_time']}-{op['end_time']}")