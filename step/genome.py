"""genome.py - Parameter container"""
import torch

class Genome:
    def __init__(self, params):
        self.params = {k: v.clone() for k, v in params.items()}
    
    def mutate(self, rate=0.01, std=0.1):
        new_params = {}
        for k, v in self.params.items():
            noise = torch.randn_like(v) * std * (torch.rand_like(v) < rate)
            new_params[k] = v + noise
        return Genome(new_params)
    
    def crossover(self, other, rate=0.5):
        new_params = {k: torch.where(torch.rand_like(v) < rate, v, other.params[k]) 
                      for k, v in self.params.items()}
        return Genome(new_params)