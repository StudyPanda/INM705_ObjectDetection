import torch
import cv2 as cv
import numpy as np
from torchvision import transforms
from collections import Counter

def get_device():
  if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
  elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available. Using Apple GPU.")
  else:
    device = torch.device("cpu")
    print("Neither CUDA nor MPS is available. Using CPU.")
  return device

def compute_iou(box1, box2):
  #inputs:
  #box1: tensor of shape (4)
  #box2: tensor of shape (4)
  x1, y1, w1, h1 = box1
  x2, y2, w2, h2 = box2
  xmin1, ymin1, xmax1, ymax1 = x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2
  xmin2, ymin2, xmax2, ymax2 = x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2

  #get intersection coordinates
  int_xmin = max(xmin1, xmin2)
  int_ymin = max(ymin1, ymin2)
  int_xmax = min(xmax1, xmax2)
  int_ymax = min(ymax1, ymax2)

  #get intersection area
  int_area = max(0, int_xmax - int_xmin) * max(0, int_ymax - int_ymin)

  #get union area
  union_area = w1*h1 + w2*h2 - int_area

  return int_area/(union_area + 1e-6)


def one_hot_encode(class_index, num_classes):
  """
  Returns a one-hot encoded list for a given class index and total number of classes.
  Args:
    class_index (int): The index of the class the object belongs to (0-based index).
    num_classes (int): The total number of classes.
  Returns:
    list: A list representing the one-hot encoded form of the class index.
  """
  if class_index >= num_classes or class_index < 0:
    raise ValueError("class_index must be within the range of total classes")

  # Create a list with all zeros
  encoding = [0] * num_classes
  # Set the element at class_index to 1
  encoding[class_index] = 1
  return encoding

def get_boxes(output, threshold = 0.25, pred=True):
  """
  Returns a list of bounding boxes with a confidence level greater than the threshold.
  Args:
    output (torch.Tensor): The output tensor from the model (N, 7, 7, num_boxes*5 + num_classes).
    threshold (float): The confidence threshold.
  Returns:
    dict: A dict of bounding boxes with a confidence level greater than the threshold. key: index, value: [[class, confidence, xmin, ymin, xmax, ymax]...]
  """
  boxes = {}
  for b in range(output.shape[0]):
    boxes[b] = []
    for i in range(7):
      for j in range(7):
        if output[b, i, j, 0] > threshold:
          x, y, w, h = output[b, i, j, 1:5]
          x = (x + i)/7
          y = (y + j)/7
          confidence = output[b, i, j, 0]
          class_label = torch.argmax(output[b, i, j, 10:])
          boxes[b].append([class_label, confidence, x, y, w, h])
          
        if pred:
          if output[b, i, j, 5] > threshold:
            x, y, w, h = output[b, i, j, 6:10]
            x = (x + i)/7
            y = (y + j)/7
            confidence = output[b, i, j, 5]
            class_label = torch.argmax(output[b, i, j, 10:])
            boxes[b].append([class_label, confidence, x, y, w, h])
  return boxes

def non_max_suppression(boxes, iou_threshold = 0.2):
  """
  Returns a list of bounding boxes after non-max suppression.
  Args:
    boxes (dict): key: img_index, value: [class, confidence, xmin, ymin, xmax, ymax]
    iou_threshold (float): The threshold used to determine true positives.
  Returns:
    dict: A dict of bounding boxes after non-max suppression. key: index, value: [class, confidence, xmin, ymin, xmax, ymax]
  """
  nms_boxes = {}
  #go through each image, sort boxes by confidence, then takes the box with highest confidence and looks at boxes in the same class with iou > threshold and removes them from the list.
  for key, val in boxes.items():
    val.sort(key = lambda x: x[1], reverse = True)
    while len(val) > 0:
      highest_conf = val.pop(0)
      if key not in nms_boxes:
        nms_boxes[key] = []
      nms_boxes[key].append(highest_conf)
      val = [detection for detection in val if not (detection[0] == highest_conf[0] and compute_iou(torch.tensor(highest_conf[2:]), torch.tensor(detection[2:])) > iou_threshold)]
  return nms_boxes


def compute_map(pred_boxes, object_boxes ,iou_threshold = 0.5, num_classes = 20):
  """
  Computes the mean average precision for a given set of predictions and ground truth.
  Args:
    pred_boxes (dict): key: img_index, value: [[class, confidence, xmin, ymin, xmax, ymax]...]
    gt (dict): key: img_index, value: [[class, confidence, xmin, ymin, xmax, ymax]...]
    iou_threshold (float): The threshold used to determine true positives.
    num_classes (int): The total number of classes.
  Returns:
    float: The mean average precision.
  """
  epsilon = 1e-6
  preds = {} #key: class, value: [[img_index, confidence, xmin, ymin, xmax, ymax]...]
  for key, val in pred_boxes.items():
    for box in val:
      class_label = int(box[0])
      if class_label not in preds:
        preds[class_label] = []
      preds[class_label].append([key] + box[1:])

  gt_boxes = {} #key: class, value: [[use_flag, img_index, confidence, xmin, ymin, xmax, ymax]...]
  for key, val in object_boxes.items():
    for box in val:
      class_label = int(box[0])
      if class_label not in gt_boxes:
        gt_boxes[class_label] = []
      gt_boxes[class_label].append([0, key] + box[1:])

  
  average_precisions = []
  for c in range(num_classes):
    if c not in gt_boxes:
      continue
    if c not in preds:
      average_precisions.append(0)
      continue

    pred = preds[c]
    gt = gt_boxes[c]
    pred.sort(key = lambda x: x[1], reverse = True)
    num_gt = len(gt)
    tp = torch.zeros(len(pred))
    fp = torch.zeros(len(pred))
    for idx,detection in enumerate(pred):
      best_iou = 0
      for i, true_box in enumerate(gt):
        if true_box[0] == 1 or true_box[1] != detection[0]:
          continue
        iou = compute_iou(torch.tensor(detection[2:]), torch.tensor(true_box[3:]))
        if iou > best_iou:
          best_iou = iou
          best_i = i
      if best_iou > iou_threshold:
        tp[idx] = 1
        gt[best_i][0] = 1
      else:
        fp[idx] = 1
    tp_cumsum = torch.cumsum(tp, dim = 0)
    fp_cumsum = torch.cumsum(fp, dim = 0)
    recall = tp_cumsum/(num_gt + epsilon)
    precision = tp_cumsum/(tp_cumsum + fp_cumsum + epsilon)
    precision = torch.cat((torch.tensor([1]), precision))
    recall = torch.cat((torch.tensor([0]), recall))
    average_precisions.append(torch.trapz(precision, recall))
  return sum(average_precisions)/len(average_precisions)


def torch_to_cv2(image):
    # Step 1: Reverse Normalize
    image = image.cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    image = (image * std) + mean
  
    image = image.permute(1, 2, 0).numpy()  # From [C, H, W] to [H, W, C]

    image = (image * 255).astype(np.uint8)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image

def visualize(image, labels):
  """
  Visualizes the image with bounding boxes.
  Args:
    image (torch.Tensor): The image tensor.
    labels (list): A list of labels. Each label is a list of [class, confidence, x, y, w, h]."""
  image = torch_to_cv2(image)
  for label in labels:
    class_label, confidence, x, y, w, h = label
    x1 = int((x - w/2) * 448)
    y1 = int((y - h/2) * 448)
    x2 = int((x + w/2) * 448)
    y2 = int((y + h/2) * 448)
    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.putText(image, str(int(class_label)), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  cv.imshow("Image", image)
  cv.waitKey(0)
  cv.destroyAllWindows()



    


