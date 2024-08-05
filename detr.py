import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s

class detr(nn.Module):
  def __init__(self, num_classes = 2, n_queries = 100):
    super().__init__()
    self.num_classes = num_classes
    self.n_queries = n_queries

    efficientnet = efficientnet_v2_s()
    self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-2])
    #freeze batchnorm layers for stability
    for module in self.feature_extractor.modules():
      if isinstance(module, nn.BatchNorm2d):
        module.eval()
        for param in module.parameters():
          param.requires_grad = False
    self.conv = nn.Sequential(nn.Conv2d(1280, 256, 1), nn.BatchNorm2d(256), nn.ReLU())
    self.transformer = nn.Transformer(256, 8, 6, 6)
    self.linear_class = nn.Linear(256, num_classes + 1) #projector for class prediction
    self.linear_bbox = nn.Linear(256, 4) #projector for bounding box prediction

    self.query_pos = nn.Parameter(torch.rand(self.n_queries, 256)) #64 queries
    self.row_embed = nn.Parameter(torch.rand(14, 256//2))
    self.col_embed = nn.Parameter(torch.rand(14, 256//2))



  def forward(self, x):
    x = self.feature_extractor(x)
    x = self.conv(x)
    batch_size, C, H,W = x.shape
    x = x.flatten(start_dim = 2)
    x = x.permute(2,0,1)
    pos = torch.cat([
      self.col_embed.unsqueeze(0).repeat(H,1,1),
      self.row_embed.unsqueeze(1).repeat(1,W,1)
    ], dim = -1).flatten(0,1).unsqueeze(1) #unsqueeze at 1 due to transformer expecting batch second 
    x = self.transformer(pos + x, self.query_pos.unsqueeze(1).repeat(1, batch_size, 1))
    c = self.linear_class(x).transpose(0,1).softmax(-1)
    b = self.linear_bbox(x).transpose(0,1).sigmoid()
    return c,b
  

x = torch.randn(2,3,448,448)
model = detr()
c,b = model(x)
print(c.shape, b.shape)