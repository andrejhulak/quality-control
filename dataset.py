import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
import torch

preprocess = v2.Compose([
    v2.Resize(256),
    # transforms.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True)
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# note for self, no idea what these normalize transforms do, but oh well

class ImageDataset(Dataset):
  def __init__(self, annotations_file, img_dir, set_type, transform=preprocess, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_labels = self.img_labels[self.img_labels['set_type'] == set_type].reset_index(drop=True)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    
    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)
        
    return image, label