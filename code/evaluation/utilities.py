import torch
import numpy as np
import matplotlib.pyplot as plt
from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss
from log.logger import LOGGER
from typing import Tuple
from pathlib import Path
from sklearn import metrics
import math
import json

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


def generate_roc_data(
  output: Path,
  truths: list,
  estimates: list
):
  '''
  Generate a JSON file for ROC data in the format suitable for DVC 
  '''
  
  fpr, tpr, thresholds = metrics.roc_curve(truths, estimates)
  nth_point = math.ceil(len(thresholds) / 2000)
  roc_points = list(zip(fpr, tpr, thresholds))[::nth_point]
  
  # Report the ROC in a structured JSON file.
  roc_file = output / 'roc.json'
  with open(roc_file, 'w') as file:
    json.dump(
      {
        "roc": [
          {"fpr": float(f), "tpr": float(t), "threshold": float(th)} for f, t, th in roc_points
        ]
      },
      file,
      indent = 4
    )
    
  LOGGER.info(f'ROC Data Saved.')
  return metrics.roc_auc_score(truths, estimates)
  
  
def generate_pr_auc(
  output: Path,
  truths: list,
  estimates: list
):
  '''
  Generate a JSON and return PR AUC score
  '''
  precision, recall, thresholds = metrics.precision_recall_curve(truths, estimates)
  nth_point = math.ceil(len(thresholds) / 2000)
  pr_points = list(zip(precision, recall, thresholds))[::nth_point]
  
  # Report the PR in a structured JSON file.
  pr_auc_curve = output / 'pr_curve.json'
  with open(pr_auc_curve, 'w') as file:
    json.dump(
      {
        "pr": [
          {"precision": float(p), "recall": float(r), "threshold": float(th)} for p, r, th in pr_points
        ]
      },
      file,
      indent = 4
    )
  return metrics.average_precision_score(truths, estimates)
