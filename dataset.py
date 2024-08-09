import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image

import matplotlib.pyplot as plt
from utils import one_hot_encode

def get_dataloader(batch_size = 64, split = 'train'):
  #label should be of size, (s, s, b*5 + c)
  #b is number of bounding boxes per cell
  #c is number of classes. Here 2
  #split is either 'train', 'val', or 'test'
  transform = transforms.Compose([
    transforms.Resize((448,448), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
  #dataset = VOCDataset_detr(transform, split)
  dataset = VOCDataset(transform, split)
  
  dataloader = DataLoader(dataset, batch_size, shuffle=True)
  return dataloader


class VOCDataset(Dataset):
  def __init__(self, transform=None, split = 'train'):
    self.root_dir = os.path.join('VOC', split)
    self.transform = transform
    self.img_files = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith('.jpg')]
    self.num_classes = 20
  
  def __len__(self):
    return len(self.img_files)

  def __getitem__(self, idx):
    img_path = self.img_files[idx]
    image = read_image(img_path)
    image = image.float() / 255
    txt_path = img_path.replace('.jpg', '.txt')
    
    with open(txt_path, 'r') as file:
      lines = file.readlines()
      label_list = []
      for line in lines:
        label_list.append([float(x) for x in line.split()])
    
    sample = {'image': image, 'label': label_list}
    
    if self.transform:
      sample['image'] = self.transform(sample['image'])
    label_grid = np.zeros((7,7,2*5 + self.num_classes))
    for label in sample['label']:
      i,j = int(label[1]*7), int(label[2]*7)
      x, y = label[1]*7 - i, label[2]*7 - j
      w, h = label[3], label[4]
      fill = [1, x, y, w, h] + [1, x, y, w, h] + one_hot_encode(int(label[0]), self.num_classes)
      label_grid[i,j,:] = fill
    return sample['image'], torch.tensor(label_grid)
  

class VOCDataset_detr(Dataset):
  def __init__(self, transform=None, split = 'train', num_queries = 100):
    self.root_dir = os.path.join('VOC', split)
    self.transform = transform
    self.img_files = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith('.jpg')]
    self.num_classes = 20
    self.num_queries = num_queries
  
  def __len__(self):
    return len(self.img_files)

  def __getitem__(self, idx):
    img_path = self.img_files[idx]
    image = read_image(img_path)
    image = image.float() / 255
    txt_path = img_path.replace('.jpg', '.txt')
    

    with open(txt_path, 'r') as file:
      lines = file.readlines()
      label_list = []
      for line in lines:
        label_list.append([float(x) for x in line.split()])
    
    sample = {'image': image, 'label': label_list}
    
    if self.transform:
      sample['image'] = self.transform(sample['image'])
    
    class_labels = []
    boxes = []
    for label in label_list:
      encoded_label = one_hot_encode(int(label[0]), self.num_classes + 1)
      class_labels.append(encoded_label)
      boxes.append([label[1], label[2], label[3], label[4]])
    
    for i in range(self.num_queries - len(boxes)):
      boxes.append([0,0,0,0])
      class_labels.append(one_hot_encode(self.num_classes, self.num_classes + 1))

    return sample['image'], torch.tensor(class_labels), torch.tensor(boxes)
  



