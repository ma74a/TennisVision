import torch
from torchvision import transforms

class Config:
    TRAIN_DIR=""
    VAL_DIR=""

    BATCH_SIZE=8
    EPOCHS=30
    LR=0.0001
    IMG_SIZE=224

    OUTPUT_POINTS=8

    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")


    COURT_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])