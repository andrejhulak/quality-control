import numpy as np
import os
from dataset import ImageDataset
from torch.utils.data import DataLoader
import torch
from vit_pytorch import ViT
from vit_pytorch.cct import CCT
from scripts.model import *

# remember that neurala did it for first 500 and first 100

if __name__ == '__main__':
  img_dir = 'casting_data'
  annotations_file = 'casting_data/annotations.csv'

  BATCH_SIZE = 32
  num_epochs = 1

  ds_train = ImageDataset(annotations_file=annotations_file, img_dir=img_dir, set_type='train')
  ds_test = ImageDataset(annotations_file=annotations_file, img_dir=img_dir, set_type='test')
  
  dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
  dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

  # image_size = 64
  # patch_size = 2
  # num_classes = 2
  # dim = 2
  # depth = 2
  # heads = 2
  # mlp_dim = 128
  # dropout = 0.1
  # emb_dropout = 0.1

  # model_config = {
  #   'image_size' : image_size,
  #   'patch_size' : patch_size,
  #   'num_classes' : num_classes,
  #   'dim' : dim,
  #   'depth' : depth,
  #   'heads' : heads,
  #   'mlp_dim' : mlp_dim,
  #   'dropout' : dropout,
  #   'emb_dropout' : emb_dropout
  #   }

  # model = ViT(image_size = image_size,
  #             patch_size = patch_size,
  #             num_classes = num_classes,
  #             dim = dim,
  #             depth = depth,
  #             heads = heads,
  #             mlp_dim = mlp_dim,
  #             dropout = dropout,
  #             emb_dropout = emb_dropout)
            
  model = CCT(
    img_size = (64, 64),
    embedding_dim = 32,
    n_conv_layers = 2,
    kernel_size = 7,
    stride = 2,
    padding = 3,
    pooling_kernel_size = 3,
    pooling_stride = 2,
    pooling_padding = 1,
    num_layers = 4,
    num_heads = 4,
    mlp_ratio = 3.,
    num_classes = 2,
    positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
)
          
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

  # y_true = 

  y_pred = test_step(model, dl_test, loss_fn=loss_fn, device='cpu')

  print(y_pred)

  # save_models(model=model, model_config=model_config, save_dir='models/test')