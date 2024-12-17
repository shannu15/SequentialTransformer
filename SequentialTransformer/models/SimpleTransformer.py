from math import gcd, ceil
import functools
import torch
from torch import nn, einsum
import torch.nn.functional as F
from models.Rotary_Embedding_torch import RotaryEmbedding, apply_rotary_emb
from einops import rearrange, repeat
import numpy as np
# -----------helper functions---------------
def exists(val):
  return val is not None
def default(val, d):
  return val if exists(val) else d

class PreNorm(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.fn = fn
    self.norm = nn.LayerNorm(dim)
  def forward(self, x, **kwargs):
    x = self.norm(x)
    return self.fn(x, **kwargs)

class FeedForward(nn.Module):
  def __init__(self, dim, mult = 4, dropout = 0.):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(dim, dim * mult),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(dim * mult, dim))
  def forward(self, x):
    return self.net(x)

class MHAttention(nn.Module):
  def __init__(
    self,
    *,
    dim = 512, # embedding size
    heads = 8,
    causal = True,
    sequence_len = 1024,
    layer_num=0,
    pos_emb = None,
    dropout = 0.,):
    super().__init__()
    self.sequence_len = sequence_len
    self.layer_num = layer_num
    self.dim_head = dim//heads
    self.scale = self.dim_head ** -0.5
    self.heads = heads
    self.causal = causal
    self.norm = nn.LayerNorm(self.dim_head)
    self.pos_emb = default(pos_emb, RotaryEmbedding(self.dim_head))
    self.attn_dropout = nn.Dropout(dropout)
    self.to_q = nn.Linear(dim, dim, bias = False)
    self.to_kv = nn.Linear(dim, dim, bias = False)
    self.to_out = nn.Linear(dim, dim)
    self.to_out_0 = nn.Linear(dim, dim)
  def forward(self, x, mask = None): # e.g., x has shape of (4,1024,256)
    b, n, *_, h, device, causal = *x.shape, self.heads, x.device, self.causal,
    mask_value = -torch.finfo(x.dtype).max
    # get queries, keys, values
    qkv = (self.to_q(x), self.to_kv(x)) # x = (4, 1024, 256)
    padded_len = x.shape[-2] # 1024
    # get sequence range, for calculating mask
    seq_range = torch.arange(padded_len, device = device)
    # split heads
    q, kv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

    # rotary embedding
    if self.layer_num == 0:
      rotary_emb = self.pos_emb(seq_range, cache_key = padded_len)
      rotary_emb = rearrange(rotary_emb, 'n d -> () n d')
      q, kv = map(lambda t: apply_rotary_emb(rotary_emb, t), (q, kv))
    q = q * self.scale # scale queries
    lkv = self.norm(kv)
    att = einsum('b m d, b n d -> b m n', q, kv)
    
    # no masking needed in classification
    
    # final attention
    attn_0= att.softmax(dim=-1)
    attnd_0 = self.attn_dropout(attn_0)
    out0 = einsum('b i j, b j d -> b i d', attnd_0, kv)
    out_0 = rearrange(out0, '(b h) n d -> b (n) (h d)', h = h)
    out_o = self.to_out_0(out_0)
    return out_o

class SimpleTransformer(nn.Module):
  def __init__(self,*,num_tokens,dim,num_layers,heads = 8,sequence_len,causal = True,ff_mult = 4,ff_dropout = 0.,attn_dropout = 0.,):
    super().__init__()
    self.sequence_len = sequence_len
    #self.token_emb = nn.Embedding(num_tokens, dim) # in NLP
    self.token_emb = nn.Linear(1,dim) # input is a single pixel
    # for color choose 3 instead of 1
    self.dim_head = dim//heads
    pos_emb = RotaryEmbedding(self.dim_head)
    self.layers = nn.ModuleList([])
    for i in range(num_layers):
      self.layers.append(nn.ModuleList([PreNorm(dim, MHAttention(dim = dim, heads = heads, sequence_len = sequence_len, layer_num=i, causal = causal, pos_emb = pos_emb, dropout = attn_dropout)),
      PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, dropout =ff_dropout))]))
    self.to_logits = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim, num_tokens))
    self.sm = nn.Softmax(dim=-1)
  def forward(self, x, mask = None):
    x = self.token_emb(x)
    for attn, ff in self.layers:
      x = attn(x, mask = mask) + x
      x = ff(x) + x
    out = self.to_logits(x)
    out2 = self.sm(out)
    return out, out2
