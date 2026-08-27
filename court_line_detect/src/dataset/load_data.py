from torch.utils.data import DataLoader

from dataset import TennisCourtDataset
from utils import Config

def load_data():
    train_dataset = TennisCourtDataset(data_dir=Config.DATA_DIR, 
                                       transforms=Config.COURT_TRANSFORMS)
    val_dataset = TennisCourtDataset(data_dir=Config.DATA_DIR, 
                                     transforms=Config.COURT_TRANSFORMS)

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=Config.BATCH_SIZE, 
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=Config.BATCH_SIZE, 
                            shuffle=False,
                            pin_memory=True)

    return train_loader, val_loader