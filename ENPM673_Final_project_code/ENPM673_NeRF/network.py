import torch
import torch.nn as nn
import torch.nn.functional as F

from ray import get_rays
from render import render
from utils import mini_batches, positional_encoding


def get_rgblogits(h, w, f, pose, near, far, Nc, batch_size, N_encode, model, device):
    # Move model to device
    model = model.to(device)

    ray_directions, ray_origins, depth_values, query_points = get_rays(
        h, w, f, pose, near, far, Nc, device
    )
    flat_query_pts = query_points.reshape((-1, 3))

    # positional encoding
    encoded_points = positional_encoding(flat_query_pts, N_encode)  # .to(device)

    batches = mini_batches(encoded_points, batch_size=batch_size)

    # Pre-calculate unflat_shape
    unflat_shape = list(query_points.shape[:-1]) + [4]

    predictions = []

    for batch in batches:
        # Move batch to device
        batch = batch.to(device)
        predictions.append(model(batch))

    radiance_field_flat = torch.cat(predictions, dim=0)
    radiance_field = torch.reshape(radiance_field_flat, unflat_shape)
    logits_rgb, _, _ = render(radiance_field, ray_directions, depth_values)

    return logits_rgb


class tNerf(nn.Module):

    def __init__(self, filter_sz=128, N_encode=6):
        super(tNerf, self).__init__()

        self.layer1 = nn.Linear(3 + 3 * 2 * N_encode, filter_sz)
        self.layer2 = nn.Linear(filter_sz, filter_sz)
        self.layer3 = nn.Linear(filter_sz, 4)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x
