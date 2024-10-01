import numpy as np
import os
from dataset import ImageDataset
from torch.utils.data import DataLoader
import torch
from vit_pytorch import ViT
from scripts.model import train

if __name__ == '__main__':

  img_dir = 'casting_data'
  annotations_file = 'casting_data/annotations.csv'

  BATCH_SIZE = 32
  num_epochs = 20

  ds_train = ImageDataset(annotations_file=annotations_file, img_dir=img_dir, set_type='train')
  ds_test = ImageDataset(annotations_file=annotations_file, img_dir=img_dir, set_type='test')
  
  dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
  dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

  model = ViT(image_size = 256,
              patch_size = 16,
              num_classes = 2,
              dim = 8,
              depth = 4,
              heads = 8,
              mlp_dim = 256,
              dropout = 0.1,
              emb_dropout = 0.1)
  
  pytorch_total_params = sum(p.numel() for p in model.parameters())
  print(pytorch_total_params)

  optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
  loss_fn = torch.nn.CrossEntropyLoss()
  
  results = train(model=model,
                  train_dataloader=dl_train,
                  test_dataloader=dl_test,
                  optimizer=optimizer,
                  loss_fn=loss_fn,
                  epochs=num_epochs,
                  device='cpu')
  print(results)