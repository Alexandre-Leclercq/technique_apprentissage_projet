import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pathlib
from typing import Tuple


class CustomDataset(Dataset):

    DATA_TYPE = torch.float32

    def __init__(self, targ_dir: str, filenames: np.array, targets: np.array, classes, img_size: int):
        self._device = torch.device('cpu')
        if torch.backends.cuda.is_built():
            self._device = torch.device('cuda')

        self.paths = [pathlib.Path(targ_dir) / (str(filename) + '.jpg') for filename in filenames]
        self.classes = classes
        self.targets = torch.tensor(targets, dtype=self.DATA_TYPE, device=self._device)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),  # Normalize for [0,1] to [-1, 1]
        ])

    def load_image(self, index: int) -> Image.Image:
        return Image.open(self.paths[index])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img: Image.Image = self.load_image(index)

        return self.transform(img).to(dtype=self.DATA_TYPE, device=self._device), self.targets[index]