import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image

import matplotlib.pyplot as plt
from utils import one_hot_encode


#get Football Dataset from directories
class FootballDatasetv1(Dataset):
  def __init__(self, data_dir = 'football_data', split = 'train', transform = None):
    super(FootballDatasetv1, self).__init__()
    if split == 'train':
      self.data_dir = os.path.join(data_dir, 'train')
    if split == 'val':
      self.data_dir = os.path.join(data_dir, 'valid')
    if split == 'test':
      self.data_dir = os.path.join(data_dir, 'test')

    self.img_dir = os.path.join(self.data_dir, 'images')
    self.label_dir = os.path.join(self.data_dir, 'labels')
    self.img_list = os.listdir(self.img_dir)

    self.transform = transform


  def __len__(self):
    return len(self.img_list)

  def __getitem__(self, idx):
    #get image at idx
    image = read_image(os.path.join(self.img_dir, self.img_list[idx]))
    image = image.float() / 255

    if self.transform:
      image = self.transform(image)

    #get labels at idx
    label_file = os.path.join(self.label_dir, self.img_list[idx][:-3] + 'txt')
    with open(label_file, 'r') as file:
      lines = file.readlines()
      label_list = []
      for line in lines:
        label_list.append([float(x) for x in line.split()])
    #visualize(image, label_list)
    label_grid = np.zeros((12,7,7)) 
    for item in label_list:
      c, x,y,w,h = item
      w *= 7
      h *= 7
      i, j = int(x*7), int(y*7)
      x, y = x*7 - i, y*7 - j

      fill = [1, x ,y, w, h] +  [1, x, y, w, h] 
      if c == 0:
        fill += [1,0]
      else:
        fill += [0,1]
      label_grid[:,i,j] = fill
    
    label_tensor = torch.tensor(label_grid)
    return image, label_tensor #label_tensor: (12,7,7). 12 = [box1, box2, class]. box is [confidence, x, y, w, h]



def get_dataloader(batch_size = 64, split = 'train'):
  #label should be of size, (s, s, b*5 + c)
  #b is number of bounding boxes per cell
  #c is number of classes. Here 2
  #split is either 'train', 'val', or 'test'
  transform = transforms.Compose([
    transforms.Resize((448,448), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
  dataset = VOCDataset(transform, split)
  #dataset = FootballDatasetv1('football_data', split, transform)
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
    
    # Here you would typically process your txt file.
    # For demonstration, let's assume it contains labels in plain text.
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
  

  






