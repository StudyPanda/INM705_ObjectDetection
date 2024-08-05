import torch
import torch.nn as nn
from utils import compute_iou

"""from utils import get_device
from dataset import get_dataloader
from torch.utils.data import DataLoader
from model import Yolov1"""



class Yolov1Loss(nn.Module):
  def __init__(self, num_boxes = 2, num_classes = 20):
    super(Yolov1Loss, self).__init__()
    self.num_boxes = num_boxes
    self.num_classes = num_classes
    self.lambda_coord = 5
    self.lambda_noobj = 0.5

  def forward(self, pred, target):
    #inputs:
    #pred: tensor of shape (batch_size, 7,7, (5*num_boxes + num_classes))
    #target: tensor of shape (batch_size, 7,7, (5*num_boxes + num_classes))
    #return loss
    batch_size = pred.shape[0]
    bbox_loss = 0
    confidence_loss = 0
    class_loss = 0

    for batch in range(batch_size):
      for i in range(7):
        for j in range(7):
          if target[batch, i, j, 0] == 1:
            class_loss += nn.functional.mse_loss(pred[batch,i,j,10:], target[batch, i,j,10:], reduction = 'sum')
            #there is object in cell i, j
            iou1 = compute_iou(pred[batch, i, j, 1:5], target[batch,i,j, 1:5])
            iou2 = compute_iou(pred[batch, i,j,6:10], target[batch,i,j, 6:10])
            if iou1 > iou2:
              bbox_loss += self.lambda_coord*nn.functional.mse_loss(pred[batch, i,j,1:3], target[batch, i,j, 1:3], reduction='sum') + self.lambda_coord*nn.functional.mse_loss(torch.sqrt(pred[batch, i, j, 3:5]+ 1e-6) , torch.sqrt(target[batch, i, j, 3:5]+ 1e-6), reduction='sum')
              confidence_loss += nn.functional.mse_loss(pred[batch, i,j,0], target[batch, i,j, 0], reduction='sum') + self.lambda_noobj*nn.functional.mse_loss(pred[batch, i,j,5], target[batch, i,j,5], reduction='sum')
            else:
              bbox_loss += self.lambda_coord*nn.functional.mse_loss(pred[batch, i,j,6:8], target[batch, i,j, 6:8], reduction='sum') + self.lambda_coord*nn.functional.mse_loss(torch.sqrt(pred[batch, i, j, 8:10] + 1e-6) , torch.sqrt(target[batch, i, j, 8:10]+ 1e-6), reduction='sum')
              confidence_loss += nn.functional.mse_loss(pred[batch, i,j,5], target[batch, i,j, 5], reduction='sum') + self.lambda_noobj*nn.functional.mse_loss(pred[batch, i,j,0], target[batch, i,j,0], reduction='sum')
          else:
            confidence_loss += self.lambda_noobj*nn.functional.mse_loss(pred[batch, i,j,0], target[batch, i,j,0], reduction='sum') + self.lambda_noobj*nn.functional.mse_loss(pred[batch, i,j,5], target[batch, i,j,5], reduction='sum')
    print(f"bbox_loss: {bbox_loss}, confidence_loss: {confidence_loss}, class_loss: {class_loss}")
    return bbox_loss + confidence_loss + class_loss