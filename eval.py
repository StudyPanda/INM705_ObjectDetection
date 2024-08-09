import torch
import torch.nn as nn

from utils import get_device, get_boxes, non_max_suppression, compute_map
from model import Yolov1
from dataset import get_dataloader
from loss import Yolov1Loss

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize(image, boxes, ground_truth):
  classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
  mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
  image = image.cpu() * std + mean
  image = image.clamp(0, 1)
  image = image.permute(1, 2, 0).numpy()
  fig, ax = plt.subplots()
  ax.imshow(image)
  for box in boxes:
    x, y, w, h = box[2:]
    x, y, w, h = int(x*448), int(y*448), int(w*448), int(h*448)
    rect = patches.Rectangle((x - w/2, y - h/2), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    class_label = classes[int(box[0])]
    ax.text(x - w/2, y - h/2, class_label, color = 'r')

  for box in ground_truth:
    x, y, w, h = box[2:]
    x, y, w, h = int(x*448), int(y*448), int(w*448), int(h*448)
    rect = patches.Rectangle((x - w/2, y - h/2), w, h, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    class_label = classes[int(box[0])]
    ax.text(x - w/2, y - h/2, class_label, color = 'g')
  plt.savefig('valid_detection2.png')
  plt.show()






device = get_device()
model = Yolov1(num_classes=20, pretrained=True)
model = model.to(device)
checkpoint = torch.load('VOC_yolov1_part2.pth')
model.load_state_dict(checkpoint['model'])

dl = get_dataloader(split = 'valid')

criterion = Yolov1Loss()
criterion = criterion.to(device)

maps = []
model.eval()
with torch.no_grad():
  for x, y in dl:
    x = x.to(torch.float32).to(device)
    y = y.to(torch.float32).to(device)
    output = model(x)
    detections = get_boxes(output, threshold=0.1)
    detections = non_max_suppression(detections, iou_threshold=0.4)
    print(type(x[0]))
    print(x[0].shape)

    ground_truth = get_boxes(y, pred=False)
    visualize(x[1], detections[1], ground_truth[1])

    break
 
      
      

  