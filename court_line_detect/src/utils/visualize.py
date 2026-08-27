import matplotlib.pyplot as plt
from typing import List
import numpy as np

def plot_losses(train_losses: List, val_losses: List) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("loss_plot.png")