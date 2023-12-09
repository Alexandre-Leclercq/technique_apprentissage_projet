import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pathlib
from typing import Tuple


class CustomDataset(Dataset):

    def __init__(self, targ_dir: str, filenames: np.array, targets, classes):
        self.paths = [pathlib.Path(targ_dir) / (str(filename) + '.jpg') for filename in filenames]
        self.classes = classes
        self.targets = targets
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)  # Normalize for [0,1] to [-1, 1]
        ])

    def load_image(self, index: int) -> Image.Image:
        return Image.open(self.paths[index])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img: Image.Image = self.load_image(index)

        return self.transform(img), self.targets[index]