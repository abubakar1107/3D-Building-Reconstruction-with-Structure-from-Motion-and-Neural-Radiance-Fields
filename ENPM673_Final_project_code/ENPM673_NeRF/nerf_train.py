import argparse
import logging
import random


import torch
import torch.nn.functional as F
from tqdm import tqdm

from network import get_rgblogits, tNerf
from utils import plot_figures, read_data

torch.manual_seed(1200)
random.seed(1200)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_PATH = "results/model.ckpt"


def train_operation(
    images,
    poses,
    focal,
    height,
    width,
    lr,
    n_encode,
    epochs,
    near_threshold,
    far_threshold,
    batch_size,
    nc,
    device,
):
    # Initiate model
    model = tNerf()
    model = model.to(device)

    # Initiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    image_count = images.shape[0]
    loss_history = []
    epochs_range = []

    logger.info(f"Training {epochs} epochs with {image_count} images ...")
    for epoch in tqdm(range(epochs)):
      
        indices = torch.randperm(image_count)

        for img_idx in tqdm(indices, desc="Processing images..."):
            img = images[img_idx].to(device)
            pose = poses[img_idx].to(device)

            rgb_logit = get_rgblogits(
                height,
                width,
                focal,
                pose,
                near_threshold,
                far_threshold,
                nc,
                batch_size,
                n_encode,
                model,
                device,
            )
            loss = F.mse_loss(rgb_logit, img)  # Photometric loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        logger.info(f"Loss at {epoch} epoch: ", loss.item())
        loss_history.append(loss.item())
        epochs_range.append(epoch + 1)

    plot_figures(epochs_range, loss_history)

    # Save the model
    with open(MODEL_PATH, "wb") as f:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
            },
            f,
        )


def detect_device():
    
    """Detects the device to use for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--nc", type=int, default=32)
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=128,
        help="Size of the mini-batch to use, default: 128",
    )
    parser.add_argument("--nn", type=int, default=2)
    parser.add_argument("--nf", type=int, default=6)

    args = parser.parse_args()

    epochs = args.num_epochs
    nc = args.nc
    batch_size = args.mini_batch_size
    near_threshold = args.nn
    far_threshold = args.nf
    device = detect_device()
    images, poses, focal = read_data()
    height, width = images.shape[1:3]
    n_encode = 6
    lr = 5e-3

    train_operation(
        images=images,
        poses=poses,
        focal=focal,
        height=height,
        width=width,
        lr=lr,
        n_encode=n_encode,
        epochs=epochs,
        near_threshold=near_threshold,
        far_threshold=far_threshold,
        batch_size=batch_size,
        nc=nc,
        device=device,
    )


if __name__ == "__main__":
    main()


# python nerf_train.py \
#     --num_epochs 10 \
#     --nc 32 \
#     --mini_batch_size 128 \
#     --nn 2 \
#     --nf 6
