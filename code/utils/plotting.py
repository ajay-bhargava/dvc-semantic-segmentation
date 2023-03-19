import matplotlib.pyplot as plt
from matplotlib import cm
import torch, torchvision
import numpy as np
from log.logger import LOGGER

def plot_ground_truths(dataloader, num_images=25):
    """
    Plots a batch of images from a PyTorch DataLoader overlaid with their predictions.

    Args:
        dataloader: PyTorch DataLoader containing the images
        num_images: number of images to plot (default=25)
    """

    # Get a batch of images
    images, masks = next(iter(dataloader))
    images = images[:num_images]
    masks = masks[:num_images]

    # Denormalize the images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    images = np.clip(images, 0, 1)

    # Standardize Images to dtype = torch.uint8 as required by draw_segmentation_masks()
    images = torchvision.transforms.functional.convert_image_dtype(images, dtype = torch.uint8) # type: ignore

    # Standardize the masks to [0,1] based on the requirements for draw_segmentation_masks()
    masks = masks / 255

    # Organize the images in a list for draw_segmentation_masks()
    overlays = [
        torchvision.utils.draw_segmentation_masks(img, masks = mask.to(torch.bool), alpha = 0.5, colors = ['green'])
        for img, mask in zip(images, masks)
    ]

    # Make a grid
    grid = torchvision.utils.make_grid(overlays, nrow=5)
    f,ax = plt.subplots(1,1, figsize=(10,10))
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis('off')
    plt.show(f)