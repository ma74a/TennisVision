import torch
from tqdm import tqdm
from utils import Config


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    for images, targets in tqdm(loader):

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(
            predictions,
            targets
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_one_epoch(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for images, targets in tqdm(loader):

            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)

            loss = criterion(
                predictions,
                targets
            )

            running_loss += loss.item()

    return running_loss / len(loader)



best_val_loss = float("inf")

def train(model,
          train_loader,
          val_loader,
          criterion,
          optimizer,
          device
        ):
    train_losses = []
    val_losses = []
    for epoch in range(Config.EPOCHS):

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{Config.EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "best_model.pth"
            )

            print("Saved best model!")

    return train_losses, val_losses