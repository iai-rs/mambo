
"""
TODO

1. add train/test split
2. add birads/mass filter (maybe)

"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from pydicom import dcmread
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from ..utils.data_utils import preprocess_scan

class RSNA_Dataset(Dataset):
    def __init__(self, csv_path='data/train.csv', images_path='data/train_images', only_birads_labeled=False, transform=None, target_transform=None):
        self.csv_path = csv_path
        self.images_path = images_path
        df = pd.read_csv(self.csv_path)

        if only_birads_labeled:
            df.dropna(subset=["BIRADS"], inplace=True)
            df.reset_index(inplace=True)

        self.data = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        patient_id = self.data.loc[index, "patient_id"]
        image_id = self.data.loc[index, "image_id"]
        image_path = os.path.join(
            self.images_path, str(patient_id), str(image_id) + ".dcm"
        )

        image = dcmread(image_path).pixel_array
        image, _ = preprocess_scan(image)
        image = image / 65536
        image = image.astype(np.float32)


        label = int(self.data.get("BIRADS").fillna(-1)[index])

        if self.target_transform:
            label = self.target_transform(label)

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    csv_path = 'sample_dataset/labels.csv'
    images_path = 'sample_dataset/images'
    transform = Compose([ToTensor(), Resize((1024, 1024)), Normalize(0.5, 1.0)])
    dataset = RSNA_Dataset(csv_path, images_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2)
    for batch_idx, (x, y) in enumerate(dataloader):
        arr = x[0].numpy()[0, :, :]
        plt.imshow(arr)
        plt.show()
