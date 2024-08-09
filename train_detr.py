import os
os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128" 

import torch
import torch.nn as nn

from detr import detr
from dataset import get_dataloader
from loss import DetrLoss
from utils import get_device
import wandb

wandb.init(project='DETR')

device = get_device()
trainloader = get_dataloader(split = 'train')
valloader = get_dataloader(split = 'valid')

model = detr(num_classes=20)
model = model.to(device)

criterion = DetrLoss()
criterion = criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 200
best_loss = float('inf')
patience = 10
for i in range(num_epochs):
  model.train()
  train_running_loss = 0
  iter = 0
  for x, y, z in trainloader:
    optimizer.zero_grad()
    x = x.to(torch.float32).to(device)
    y = y.to(torch.float32).to(device)
    z = z.to(torch.float32).to(device)
    gts = {'class': y, 'boxes': z}

    output = model(x)
    loss = criterion(output, gts)
    print(f"Epoch {i}, Loss: {loss.item()}, percentage complete: {iter/len(trainloader)*100}")
    loss.backward()
    optimizer.step()
    train_running_loss += loss.item()
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'VOC_detr.pth')
    iter += 1
  val_running_loss = 0
  for x, y, z in valloader:
    x = x.to(torch.float32).to(device)
    y = y.to(torch.float32).to(device)
    z = z.to(torch.float32).to(device)
    gts = {'class': y, 'boxes': z}
    output = model(x)
    loss = criterion(output, gts)
    val_running_loss += loss.item()
  wandb.log({"Train Loss": train_running_loss/len(trainloader), "Validation Loss": val_running_loss/len(valloader)})
  print(f"Epoch {i}, Loss: {train_running_loss/len(trainloader)}, Validation Loss: {val_running_loss/len(valloader)}")
  if val_running_loss < best_loss:
    best_loss = val_running_loss
    counter = 0
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'VOC_detr_val.pth')
  else:
    counter += 1
    if counter == patience:
      break

  
