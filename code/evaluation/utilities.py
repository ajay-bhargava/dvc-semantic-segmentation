import torch
import numpy as np
import matplotlib.pyplot as plt
from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss
from log.logger import LOGGER
from typing import Tuple

def evaluate_selection(
  _image: torch.Tensor,
  _mask: torch.Tensor,
  _prediction: torch.Tensor,
) -> Tuple[float, float]: 
  '''
  Takes a batch of images, masks, and predictions and saves a single image from the batch
  '''
  # Random Generator
  random = np.random.randint(0, _image.size(0))
  
  # Compute a probability for prediction 
  prediction = torch.nn.Sigmoid()(_prediction[random])
  
  # Denormalize the images
  mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
  images = _image * std + mean
  images = np.clip(images, 0, 1)
  
  # Calculate the image, mask, and prediction
  image = images[random].permute(1,2,0).cpu().numpy()
  mask = _mask[random].cpu().numpy()
  prediction = prediction.squeeze(0).cpu().numpy()
  
  f,ax = plt.subplots(1, 3)
  ax[0].imshow(image)
  ax[0].set_title('Image')
  ax[1].imshow(mask)
  ax[1].set_title('Mask')
  ax[2].imshow(prediction)
  ax[2].set_title('Prediction')
  for n in range(3):
    ax[n].axis('off')
  f.savefig('../metrics/figures/evaluation.png', bbox_inches='tight', dpi = 300)
  plt.close(f)
  
  # Calculate metrics for single mask versus prediction.
  _prediction_ = torch.nn.Sigmoid()(_prediction[random]) 
  jaccard = 1. - JaccardLoss(mode = 'binary', from_logits = False)(_prediction_.unsqueeze(0), _mask[random].unsqueeze(0)).detach().cpu().numpy()
  dice = 1. - DiceLoss(mode = 'binary', from_logits = False)(_prediction_.unsqueeze(0), _mask[random].unsqueeze(0)).detach().cpu().numpy()
  return jaccard, dice