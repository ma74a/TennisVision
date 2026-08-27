import torch

from dataset.load_data import load_data
from model import TennisCourtModel
from utils import Config
from  utils.visualize import plot_losses
from training import train


def main():
    train_loader, val_loader = load_data()

    model = TennisCourtModel(output_points=Config.OUTPUT_POINTS)
    model.to(Config.DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)

    train_losses, val_losses = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=Config.DEVICE
        )

    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()