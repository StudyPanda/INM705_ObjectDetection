import torch
import torch.nn as nn

from utils import get_device, get_boxes, non_max_suppression, compute_map
from dataset import get_dataloader
from torch.utils.data import DataLoader
from model import Yolov1
from loss import Yolov1Loss

import wandb

wandb.init(project='Yolov1_VOC_Final')

load_model = True
device = get_device()
trainloader = get_dataloader(split = 'train')
valloader = get_dataloader(split = 'valid')

model = Yolov1(num_classes=20, pretrained=True)
model = model.to(device)

criterion = Yolov1Loss()
criterion = criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
if load_model:
    checkpoint = torch.load('VOC_yolov1.pth')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

epochs = 75
best_map = 0
counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0
    iter = 0
    for x, y in trainloader:
        optimizer.zero_grad()
        x = x.to(torch.float32).to(device)
        y = y.to(torch.float32).to(device)
        output = model(x)
        loss = criterion(output, y)
        if torch.isnan(loss).any():
            print(f"NaN detected at iteration {iter}")
            print("Gradients:", [x.grad for x in model.parameters() if x.grad is not None]) 
            break
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        iter += 1
        print(f"Epoch:{epoch}, {iter/len(trainloader)*100}% complete, last loss: {loss.item()}") 
    print(f"Epoch: {epoch}, Loss: {running_loss/len(trainloader)}")
    wandb.log({"Train Loss": running_loss/len(trainloader)})

    #validation
    model.eval()
    with torch.no_grad():
        running_loss = 0
        for x, y in valloader:
            x = x.to(torch.float32).to(device)
            y = y.to(torch.float32).to(device)
            output = model(x)
            loss = criterion(output, y)
            running_loss += loss.item()
        print(f"Validation Loss: {running_loss/len(valloader)}")
        wandb.log({"Validation Loss": running_loss/len(valloader)})      
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, "VOC_yolov1_part2.pth")
        counter = 0

wandb.finish()

