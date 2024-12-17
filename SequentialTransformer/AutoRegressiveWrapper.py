import torch
from torch import nn
import torch.nn.functional as F
class AutoRegressiveWrapper(nn.Module):
  def __init__(self, net):
    super().__init__()
    self.model = net
    self.max_seq_len = net.sequence_len
  
  def forward(self, x, target):
    xi = x # x is input of size seq_len
    out, out2 = self.model(xi)
    out = out[:,-1,:] # last output to be used for classification
    logits_reorg = out.reshape(-1, out.size(-1))
    targets_reorg = target.reshape(-1)
    loss = F.cross_entropy(logits_reorg, targets_reorg)
    return loss, out2 # out2 is softmax output