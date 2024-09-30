import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
from moviepy.editor import ImageSequenceClip


def read_data():
    data = np.load("data/lego/tiny_nerf_data.npz")
    images = torch.from_numpy(data["images"])
    poses = torch.from_numpy(data["poses"])
    focal = torch.from_numpy(data["focal"])

    return images, poses, focal


def positional_encoding(flat_query_pts, N_encode):
    gamma = [flat_query_pts]
    for i in range(N_encode):
        gamma.append(torch.sin((2.0**i) * flat_query_pts))
        gamma.append(torch.cos((2.0**i) * flat_query_pts))

    gamma = torch.cat(gamma, dim=-1)

    return gamma


def mini_batches(inputs, batch_size):
    return [inputs[i : i + batch_size] for i in range(0, inputs.shape[0], batch_size)]


def plot_figures(Epochs, log_loss):
    plt.figure(figsize=(10, 4))
    plt.plot(Epochs, log_loss)
    plt.title("Loss")
    plt.savefig("results/loss_epochs_plot.png")


def make_video(fps, path, video_file):

    clip = ImageSequenceClip(path, fps=fps)
    clip.write_videofile(video_file)
    shutil.rmtree(path)
