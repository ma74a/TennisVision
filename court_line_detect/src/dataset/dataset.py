import torch
from torch.utils.data import Dataset

import os
from PIL import Image


class TennisCourtDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.images_dir = os.path.join(self.data_dir, "images")
        self.labels_dir = os.path.join(self.data_dir, "labels")
        self.transforms = transforms
        self.images = []
        self.labels = []

        exclueded = [n for n in range(138,144)]
        for i, img_name in enumerate(sorted(os.listdir(self.images_dir))):
            if i in exclueded:
                continue
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(self.images_dir, img_name)

            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(self.labels_dir, label_name)

            # Skip if label doesn't exist
            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                line = f.readline()


            lines = line.strip().split()
            values = list(map(float, lines[5:]))
            key_points_lst = []
            not_included = [2, 5, 8, 11]
            for i, val in enumerate(values):
                if i in not_included:
                    continue
                key_points_lst.append(val)
            matrix = [key_points_lst[i:i+2] for i in range(0, len(key_points_lst), 2)]

            self.images.append(img_path)
            self.labels.append(torch.tensor(matrix, dtype=torch.float32))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        img = Image.open(img_path).convert("RGB")

        if self.transforms:
            img = self.transforms(img)
        
        return img, label

