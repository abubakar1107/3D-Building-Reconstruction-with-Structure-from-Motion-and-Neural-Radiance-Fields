import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.transform import resize

from network import get_rgblogits, tNerf
from utils import make_video

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

JSON_PATH = Path("data/lego/transforms_test.json")
DATASET_PATH = "data/lego/"
MODEL_PATH = Path("results/model.ckpt")


def read_json(jsonPath):
    # open the json file
    with open(jsonPath, "r") as fp:
        # read the json data
        data = json.load(fp)

    # return the data
    return data


def get_image_c2w(jsonData, datasetPath):
    # define a list to store the image paths
    imagePaths = []

    # define a list to store the camera2world matrices
    c2ws = [] 
    # iterate over each frame of the data
    for frame in jsonData["frames"]:
        # grab the image file name
        imagePath = frame["file_path"]
        imagePath = imagePath.replace(".", datasetPath)
        imagePaths.append(f"{imagePath}.png")

        # grab the camera2world matrix
        c2ws.append(frame["transform_matrix"])

    # return the image file names and the camera2world matrices
    return imagePaths, c2ws


def read_images(imagePaths):
    images = []
    for i in range(len(imagePaths)):
        image = plt.imread(imagePaths[i])
        image.resize((100, 100, 3))
        images.append(image)
    images = np.array(images)
    images = torch.from_numpy(images)
    return images

def test_operation(
    images,
    poses,
    focal,
    height,
    width,
    lr,
    N_encode,
    near_threshold,
    far_threshold,
    batch_size,
    Nc,
    device,
    ModelPath,
    save_path,
):

    model = tNerf()
    checkpoint = torch.load(ModelPath)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(torch.float64)
    model = model.to(device)
    print(device)

    count = 1
    model.eval()

    logger.info(f"Testing the model on {len(images)} images...")
    loss_history = []
    for i in tqdm(range(len(images))):
        img = images[i].to(device)
        pose = poses[i].to(device)

        rgb_logit = get_rgblogits(
            height,
            width,
            focal,
            pose,
            near_threshold,
            far_threshold,
            Nc,
            batch_size,
            N_encode,
            model,
            device,
        )

        # Ensure the shapes of img and rgb_logit are compatible
        if img.shape != rgb_logit.shape:
            logger.error(f"Shape mismatch for image {i}: img.shape={img.shape}, rgb_logit.shape={rgb_logit.shape}")
            continue

        loss = F.mse_loss(rgb_logit, img)

        loss_history.append(loss.item())

        # Convert the tensor to a numpy array
        image_data = rgb_logit.detach().cpu().numpy()

        # Resize the image 
        resized_image = resize(image_data, (500, 500))

        # Normalize the image data to 0-255 range
        resized_image = (resized_image * 255).astype(np.uint8)

        # Save the resized image
        imageio.imsave(save_path + "/" + str(count) + ".png", resized_image)

        count += 1

        # torch.cuda.empty_cache()

    if len(loss_history) == 0:
        logger.error("No valid losses computed.")
    else:
        avg_loss = sum(loss_history) / len(loss_history)
        logger.info(f"Average loss on test data: {avg_loss}")

    return avg_loss


def detect_device():
    """
    Detect the device to load the model on.

    Returns:
        str: The device to load the model on.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=int, default=32)
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=1,
        help="Size of the MiniBatch to use, Default:1",
    )
    parser.add_argument("--nn", type=int, default=2)
    parser.add_argument("--nf", type=int, default=6)
    return parser.parse_args()


def main():

    save_path = os.path.join("test_results")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    else:
        os.mkdir(save_path)

    # device = torch.device("cpu")  # detect_device()
    device = detect_device()

    # load the data file
    data = read_json(JSON_PATH)

    imagePaths, poses = get_image_c2w(data, DATASET_PATH)
    poses = np.array(poses)
    poses = torch.from_numpy(poses)

    # load the images
    images = read_images(imagePaths)

    ModelPath = "results/model.ckpt"

    args = parse_args()

    nc = args.nc
    batch_size = args.mini_batch_size
    near_threshold = args.nn
    far_threshold = args.nf

    focal = np.array([138.8889])
    focal = torch.from_numpy(focal).to(device)
    height, width = images.shape[1:3]

    N_encode = 6
    lr = 5e-3

    logger.info("Initiating testing ...")
    test_operation(
        images,
        poses,
        focal,
        height,
        width,
        lr,
        N_encode,
        near_threshold,
        far_threshold,
        batch_size,
        nc,
        device,
        ModelPath,
        save_path,
    )

    video_file = "NerF.mp4"
    fps = 5
    logger.info(f"Creating video {video_file}, FPS={fps} ...")
    make_video(fps, save_path, video_file)


if __name__ == "__main__":
    main()
