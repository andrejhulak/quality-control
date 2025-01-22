import numpy as np
import os
from dataset import ImageDataset
from torch.utils.data import DataLoader
import torch
from vit_pytorch import ViT
from vit_pytorch.cct import CCT
from scripts.model import *
from vit_pytorch.recorder import Recorder
from vit_pytorch.extractor import Extractor

# use segmentation techniques
# how???

if __name__ == '__main__':
  img_dir = 'linsen_data'
  annotations_file = 'linsen_data/annotations.csv'

  BATCH_SIZE = 32
  num_epochs = 5

  ds_train = ImageDataset(annotations_file=annotations_file, img_dir=img_dir, set_type='train')
  ds_test = ImageDataset(annotations_file=annotations_file, img_dir=img_dir, set_type='test')
  
  dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
  dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

  model_config = {
    'img_size' : (128, 128),
    'embedding_dim' : 32,
    'n_conv_layers' : 2,
    'kernel_size' : 7,
    'stride' : 2,
    'padding' : 3,
    'pooling_kernel_size' : 3,
    'pooling_stride' : 2,
    'pooling_padding' : 1,
    'num_layers' : 4,
    'num_heads' : 4,
    'mlp_ratio' : 3.,
    'num_classes' : 2,
    'positional_embedding' : 'learnable'
    }

  model = CCT(
    img_size = model_config['img_size'],
    embedding_dim = model_config['embedding_dim'],
    n_conv_layers = model_config['n_conv_layers'],
    kernel_size = model_config['kernel_size'],
    stride = model_config['stride'],
    padding = model_config['padding'],
    pooling_kernel_size = model_config['pooling_kernel_size'],
    pooling_stride = model_config['pooling_stride'],
    pooling_padding = model_config['pooling_padding'],
    num_layers = model_config['num_layers'],
    num_heads = model_config['num_heads'],
    mlp_ratio = model_config['mlp_ratio'],
    num_classes = model_config['num_classes'],
    positional_embedding = model_config['positional_embedding']
  )

  model.load_state_dict(torch.load('models/vit/model.pth', weights_only=True))
          
  pytorch_total_params = sum(p.numel() for p in model.parameters())
  print(pytorch_total_params)

  optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001)
  loss_fn = torch.nn.CrossEntropyLoss()

  accuracy = test_step(model, dl_test, loss_fn, 'cpu')[1]

  print(f'Accuracy: {accuracy:.4f}')
  
  # results = train(model=model,
  #                 train_dataloader=dl_train,
  #                 test_dataloader=dl_test,
  #                 optimizer=optimizer,
  #                 loss_fn=loss_fn,
  #                 epochs=num_epochs,
  #                 device='cpu')

  # save_model(model=model, model_config=model_config, save_dir='models/vit2')