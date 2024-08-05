import torch
import torch.nn as nn

from utils import get_device, get_boxes, non_max_suppression,visualize, compute_map
from model import Yolov1
from dataset import get_dataloader
from loss import Yolov1Loss


device = get_device()
model = Yolov1(num_classes=20, pretrained=True)
model = model.to(device)
checkpoint = torch.load('VOC_yolov1_part2.pth')
model.load_state_dict(checkpoint['model'])

dl = get_dataloader(split = 'valid')

criterion = Yolov1Loss()
criterion = criterion.to(device)


model.eval()
with torch.no_grad():
  for x, y in dl:
    idx = 30 
    x = x.to(torch.float32).to(device)
    y = y.to(torch.float32).to(device)
    output = model(x)
    detections = get_boxes(output, threshold=0.1)
    detections = non_max_suppression(detections, iou_threshold=0.3)
    ground_truth = get_boxes(y, pred=False)
    visualize(x[idx], detections[idx])
    visualize(x[idx], ground_truth[idx])
    break
 
      
      

  