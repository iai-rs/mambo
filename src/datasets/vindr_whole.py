import os
import matplotlib.pyplot as plt
import pandas as pd
from pydicom import dcmread
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from ..utils.data_utils import preprocess_scan


class VINDR_Dataset(Dataset):
    def __init__(self, csv_path='data/train.csv', images_path='data/train_images', transform=None):
        self.csv_path = csv_path
        self.images_path = images_path

        df = pd.read_csv(self.csv_path)
#         df = df[df["finding_categories"] == r"['No Finding']"]
#         df = df[df['breast_birads'].isin(['BI-RADS 1', 'BI-RADS 2'])] #keep only the healthiest breasts
#         df = df[df["height"] == 3518]
        df = df[df['split'] == 'training']
        df = df.reset_index(drop=True)
        self.data = df
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        study_id = self.data.loc[index, "study_id"]
        image_id = self.data.loc[index, "image_id"]
        image_path = os.path.join(
            self.images_path, str(study_id), str(image_id) + ".dicom"
        )

        image = dcmread(image_path).pixel_array
        image, _ = preprocess_scan(image)
        image = image / 65536
        image = image.astype(np.float32)

        image = self.transform(image)
        return image


if __name__ == '__main__':
    transform = Compose([ToTensor(), Resize((256, 256), antialias=True), Normalize(0.007, 0.01)])
    csv_path = '/data/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/finding_annotations.csv'
    images_path = '/data/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images/'
    dataset = VINDR_Dataset(csv_path, images_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1)
    for batch_idx, x in enumerate(dataloader):
        arr = x[0].numpy()[0, :, :]
        plt.imshow(arr)
        plt.show()
