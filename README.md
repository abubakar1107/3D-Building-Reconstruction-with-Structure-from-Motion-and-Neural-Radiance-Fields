# 3D Building Reconstruction with Structure from Motion and Neural Radiance Fields

This project implements a pipeline for 3D reconstruction of buildings using Structure from Motion (SfM) and Implementing Neural Radiance Fields (NeRF). The goal is to accurately reconstruct 3D building environments from 2D images, leveraging SfM to estimate camera poses and utilizing NeRF to render high-fidelity 3D models of objects. The project compares traditional computer vision techniques with cutting-edge deep learning methods for reconstructing buildings from photographs amd synthesizing novel views of objects.

## Features
- **Structure from Motion (SfM)**: Uses multi-view geometry to estimate camera poses and sparse point clouds from multiple 2D images.
- **Neural Radiance Fields (NeRF)**: A neural rendering technique to generate realistic 3D models from a collection of 2D views, using neural networks.
- **3D Model Reconstruction**: The combination of SfM and NeRF allows for creating high-quality 3D models of buildings from images taken at various angles.

## Requirements
To run the project, ensure you have the following installed:
- Python 3.8+
- PyTorch
- OpenCV
- COLMAP (for SfM)
- NVIDIA GPU (for NeRF training)

Install dependencies using:
```bash
pip install -r requirements.txt
