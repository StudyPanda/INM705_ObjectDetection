import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
from torchvision.ops import generalized_box_iou_loss
from loss import DetrLoss

class detr(nn.Module):
  def __init__(self, num_classes = 20, n_queries = 100):
    super().__init__()
    self.num_classes = num_classes
    self.n_queries = n_queries
    self.num_heads = 4
    self.num_encoder_layers = 2
    self.num_decoder_layers = 2
    self.embed_dim = 128

    efficientnet = efficientnet_v2_s(weights='DEFAULT')
    self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-2])
    #freeze batchnorm layers for stability
    for module in self.feature_extractor.modules():
      if isinstance(module, nn.BatchNorm2d):
        module.eval()
        for param in module.parameters():
          param.requires_grad = False
    self.conv = nn.Sequential(nn.Conv2d(1280, self.embed_dim, 1), nn.BatchNorm2d(self.embed_dim), nn.ReLU())
    self.transformer = nn.Transformer(self.embed_dim, self.num_heads, self.num_encoder_layers, self.num_decoder_layers)
    self.linear_class = nn.Linear(self.embed_dim, num_classes + 1) #projector for class prediction
    self.linear_bbox = nn.Linear(self.embed_dim, 4) #projector for bounding box prediction

    self.query_pos = nn.Parameter(torch.rand(self.n_queries, self.embed_dim)) #64 queries
    self.row_embed = nn.Parameter(torch.rand(14, self.embed_dim//2))
    self.col_embed = nn.Parameter(torch.rand(14, self.embed_dim//2))



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
    out = {}
    out['class'] = self.linear_class(x).transpose(0,1).softmax(-1)
    out['boxes'] = self.linear_bbox(x).transpose(0,1).sigmoid()
    return out
  
