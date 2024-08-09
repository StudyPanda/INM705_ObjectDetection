import torch
import torch.nn as nn
import numpy as np
from utils import compute_iou
from torchvision.ops import generalized_box_iou_loss
from scipy.optimize import linear_sum_assignment

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

class DetrLoss(nn.Module):
  def __init__(self, lambda_L1 = 1, lambda_iou = 1):
    super().__init__()
    self.lamdba_L1 = lambda_L1
    self.lambda_iou = lambda_iou

  def box_loss(self, pred_box, gt_box):
    """
    (supports any batch size)
    Inputs:
    pred_box: tensor of shape (4)
    gt_box: tensor of shape (4)
    Output:
    Loss value after performing L1 loss between the predicted and ground truth boxes.
    """
    if pred_box.dim() == 1:
      pred_box = pred_box.unsqueeze(0)
      gt_box = gt_box.unsqueeze(0)

    formatted_pred_box = torch.cat([pred_box[:,:2] - pred_box[:,2:]/2, pred_box[:,:2] + pred_box[:,2:]/2], dim = -1)
    formatted_gt_box = torch.cat([gt_box[:,:2] - gt_box[:,2:]/2, gt_box[:,:2] + gt_box[:,2:]/2], dim = -1)
    return self.lamdba_L1 * nn.functional.l1_loss(pred_box, gt_box, reduction='sum') + self.lambda_iou * generalized_box_iou_loss(formatted_pred_box, formatted_gt_box, reduction='sum')
  
  def L_match(self, pred_class, pred_box, gt_class, gt_box):
    """
    Inputs:
    pred_class: tensor of shape (num_classes + 1)
    pred_box: tensor of shape (4)
    gt: tensor of shape (num_classes + 1)
    gt_box: tensor of shape (4)
    Output:
    Loss value after performing minimal cost matching between the predicted and ground truth boxes.
    """
    indicator = 1 - gt_class[-1]
    loss = indicator*(pred_class[gt_class.argmax()] + self.box_loss(pred_box, gt_box))
    return loss

  @torch.no_grad()
  def hungarian_matcher(self, preds, ground_truths):
    """
    Inputs:
    preds and ground_truths: dictionary containing two keys: 'class' and 'boxes'
    ['class']: tensor of shape (batch_size, num_queries, num_classes + 1)
    ['boxes']: tensor of shape (batch_size, num_queries, 4)
    Output:
    List of [pred_idxs, gt_idxs] where the ith element of the list contains the minimal matching of the ith element of the batch.
    pred_idxs: List of indices of the predicted boxes
    gt_idxs: List of indices of the ground truth boxes
    the elements of pred_idxs and gt_idxs are in the same order.
    """ 
    out = []
    batch_size, num_queries, _ = preds['class'].shape
    for b in range(batch_size):
      pred_class = preds['class'][b]
      pred_boxes = preds['boxes'][b]
      gt_class = ground_truths['class'][b]
      gt_boxes = ground_truths['boxes'][b]
      loss_matrix = np.zeros((num_queries, num_queries))
      #create loss matrix
      for i in range(num_queries):
        for j in range(num_queries):
          loss_matrix[i,j] = self.L_match(pred_class[i], pred_boxes[i], gt_class[j], gt_boxes[j])
      pred_idxs, gt_idxs = linear_sum_assignment(loss_matrix)
      out.append([pred_idxs, gt_idxs])
    return out
      
  def forward(self, preds, ground_truths):
    """
    Inputs:
    preds and ground_truths: dictionary containing two keys: 'class' and 'boxes'
    ['class']: tensor of shape (batch_size, num_queries, num_classes + 1)
    ['boxes']: tensor of shape (batch_size, num_queries, 4)
    Output:
    Loss value after performing minimal cost matching between the predicted and ground truth boxes.
    """
    batch_size, num_queries, _ = preds['class'].shape
    matching = self.hungarian_matcher(preds, ground_truths)
    loss = 0
    for b in range(batch_size):
      pred_idxs, gt_idxs = matching[b]
      class_loss = nn.CrossEntropyLoss()(preds['class'][b][pred_idxs], ground_truths['class'][b][gt_idxs])
      box_loss = self.box_loss(preds['boxes'][b][pred_idxs], ground_truths['boxes'][b][gt_idxs])
      loss += class_loss + box_loss
    return loss
