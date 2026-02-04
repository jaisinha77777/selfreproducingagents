"""fitness.py - Evaluation helpers"""
import torch

def supervised_fitness(network, dataloader, criterion, device='cpu'):
    """Evaluate network on supervised task. Returns negative loss."""
    network.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            loss = criterion(network(x), y)
            total_loss += loss.item()
    return -total_loss / len(dataloader)
