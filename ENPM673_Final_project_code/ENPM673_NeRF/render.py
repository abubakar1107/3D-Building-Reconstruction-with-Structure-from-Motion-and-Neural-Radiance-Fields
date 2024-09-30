import torch
import torch.nn.functional as F


def render(radiance_field, ray_origins, depth_values):

  sigma_a = F.relu(radiance_field[...,3])       #volume density
  rgb = torch.sigmoid(radiance_field[...,:3])    #color value at nth depth value
  one_e_10 = torch.tensor([1e10], dtype = ray_origins.dtype, device = ray_origins.device)
  dists = torch.cat((depth_values[...,1:] - depth_values[...,:-1], one_e_10.expand(depth_values[...,:1].shape)), dim = -1)

  alpha = 1. - torch.exp(-sigma_a * dists)       
  weights = alpha * cumulative_prod_exclusive(1. - alpha + 1e-10)     #transmittance
  rgb_map = (weights[..., None] * rgb).sum(dim = -2)          #resultant rgb color of n depth values
  depth_map = (weights * depth_values).sum(dim = -1)
  acc_map = weights.sum(-1)

  return rgb_map, depth_map, acc_map

# To calculate the cumulative product used to calculate alpha
def cumulative_prod_exclusive(tensor) :
  dim = -1
  cumprod = torch.cumprod(tensor, dim)
  cumprod = torch.roll(cumprod, 1, dim)
  cumprod[..., 0] = 1.
  
  return cumprod