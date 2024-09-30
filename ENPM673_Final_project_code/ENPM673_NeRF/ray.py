import torch


def get_rays(h, w, f, pose, near, far, Nc, device):

    # making meshgrid
    x = torch.linspace(0, w - 1, w)
    y = torch.linspace(0, h - 1, h)

    xi, yi = torch.meshgrid(x, y, indexing="xy")
    xi = xi.to(device)
    yi = yi.to(device)

    # normalized coordinates
    norm_x = (xi - w * 0.5) / f
    norm_y = (yi - h * 0.5) / f

    # direction unit vectors matrix
    directions = torch.stack([norm_x, -norm_y, -torch.ones_like(xi)], dim=-1)
    directions = directions[..., None, :]

    # camera matrix : 3x3 matrix from the 4x4 projection matrix
    rotation = pose[:3, :3]
    translation = pose[:3, -1]

    camera_directions = directions * rotation
    ray_directions = torch.sum(camera_directions, dim=-1)
    ray_directions = ray_directions / torch.linalg.norm(
        ray_directions, dim=-1, keepdims=True
    )
    ray_origins = torch.broadcast_to(translation, ray_directions.shape)

    # get the sample points
    depth_val = torch.linspace(near, far, Nc)

    noise_shape = list(ray_origins.shape[:-1]) + [Nc]
    noise = torch.rand(size=noise_shape) * (far - near) / Nc

    depth_val = depth_val + noise
    depth_val = depth_val.to(device)

    query_points = (
        ray_origins[..., None, :]
        + ray_directions[..., None, :] * depth_val[..., :, None]
    )

    return ray_directions, ray_origins, depth_val, query_points
