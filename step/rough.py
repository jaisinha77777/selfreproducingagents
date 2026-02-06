"""example_save_load.py - Demo of saving/loading genomes and networks"""
import torch
import torch.nn as nn
from evolvable import EvolvableModule
from genome import Genome

# Define network
class SimpleNet(EvolvableModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)

print("="*60)
print("GENOME SAVE/LOAD")
print("="*60)

# Create and mutate genome
net = SimpleNet()
genome = Genome(net.state_dict())
print(f"Original genome: {genome.params['fc.weight'][0, 0].item():.4f}")

# Save genome
genome.save('genome.pkl')
print("✓ Saved genome to genome.pkl")

# Load genome
loaded_genome = Genome.load('genome.pkl')
print(f"Loaded genome:  {loaded_genome.params['fc.weight'][0, 0].item():.4f}")
print(f"Match: {torch.allclose(genome.params['fc.weight'], loaded_genome.params['fc.weight'])}")

print("\n" + "="*60)
print("NETWORK SAVE/LOAD")
print("="*60)

# Create and train a bit
net = SimpleNet()
net.fc.weight.data.fill_(1.0)  # Set to 1 for easy verification
print(f"Original network weight: {net.fc.weight[0, 0].item():.4f}")

# Save network
net.save('network.pkl')
print("✓ Saved network to network.pkl")

# Load network
loaded_net = SimpleNet.load('network.pkl')
print(f"Loaded network weight:  {loaded_net.fc.weight[0, 0].item():.4f}")
print(f"Match: {torch.allclose(net.fc.weight, loaded_net.fc.weight)}")

print("\n" + "="*60)
print("EVOLUTION WITH CHECKPOINTING")
print("="*60)

# Evolve and save best
best_net = SimpleNet()
best_fitness = -999

for gen in range(5):
    child = best_net.reproduce(mutation_rate=0.1)
    
    # Dummy fitness (just random for demo)
    fitness = torch.randn(1).item()
    
    if fitness > best_fitness:
        best_fitness = fitness
        best_net = child
        best_net.save(f'best_gen{gen}.pkl')
        print(f"Gen {gen}: New best! Fitness={fitness:.3f}, saved to best_gen{gen}.pkl")

print("\n✓ All checkpoints saved!")
print("\nTo resume evolution:")
print("  net = SimpleNet.load('best_gen4.pkl')")
print("  # Continue evolving from there...")