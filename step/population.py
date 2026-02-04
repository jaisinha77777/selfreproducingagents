"""population.py - Evolution manager"""
import torch

class Population:
    def __init__(self, network_class, size=100):
        self.individuals = [network_class() for _ in range(size)]
        self.size = size
    
    def evolve(self, fitnesses, elite_frac=0.1, mutation_rate=0.01):
        # Sort by fitness (higher is better)
        sorted_idx = torch.argsort(fitnesses, descending=True)
        
        # Keep elite
        n_elite = int(self.size * elite_frac)
        new_pop = [self.individuals[i] for i in sorted_idx[:n_elite]]
        
        # Reproduce rest via tournament selection
        while len(new_pop) < self.size:
            i, j = torch.randint(len(self.individuals), (2,))
            parent = self.individuals[i] if fitnesses[i] > fitnesses[j] else self.individuals[j]
            new_pop.append(parent.reproduce(mutation_rate))
        
        self.individuals = new_pop