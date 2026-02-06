"""evolvable.py - Self-reproducing networks"""
import torch.nn as nn
from genome import Genome

class EvolvableModule(nn.Module):
    def reproduce(self, mutation_rate=0.01):
        genome = Genome(self.state_dict())
        child_genome = genome.mutate(mutation_rate)
        child = self.__class__()
        child.load_state_dict(child_genome.params)
        return child
    
    def crossover(self, other, rate=0.5):
        genome_a = Genome(self.state_dict())
        genome_b = Genome(other.state_dict())
        child_genome = genome_a.crossover(genome_b, rate)
        child = self.__class__()
        child.load_state_dict(child_genome.params)
        return child