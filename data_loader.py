import torch
import numpy as np
from PIL import Image
import numpy as np
from torchvision import transforms

from torch.utils.data import Dataset


def convert_and_normalize(file_name, size, train=True):
    if ".npy" in file_name:
        input_image = np.load(file_name)
        input_image = transforms.ToPILImage()(input_image)
    else:
        try:
            input_image = Image.open(file_name).convert("RGB")
        except:
            print(file_name)

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.24703223, 0.24348512, 0.26158784],
    )

    if train:
        preprocess = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomCrop(32, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        preprocess = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    tensor = preprocess(input_image)
    return tensor


class DataReader(Dataset):
    def __init__(self, file_path, train=True, size=128):

        super(Dataset, self).__init__()
        self.train = train
        with open(file_path, "r") as f:
            files = f.read()
        files = sorted([f.split(" ") for f in files.split("\n") if len(f) > 1])
        self.dataset_list = sorted(files)

        self.length = len(self.dataset_list)
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            file_path, label = self.dataset_list[idx]
        except:
            print(idx, len(self.dataset_list), self.dataset_list[idx])
        label = int(label)
        image_tensor = convert_and_normalize(
            file_path, size=self.size, train=self.train
        )
        return image_tensor, torch.tensor(label, dtype=torch.long)