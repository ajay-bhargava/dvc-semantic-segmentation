'''
eval.py

Outputs: 
1. Example random inference image side
2. Metrics for that image
3. Metrics for the entire dataset.
 
Metrics include: JaccardLoss, DiceLoss 
Outputs are fed into DVC reports

This script is ready to be added to a DVC pipeline as a stage.
'''

import torch
from tqdm import tqdm
from log.logger import LOGGER
from pathlib import Path
from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss
from model.reload_model import reload_model
from loaders.prepare import prepare_loaders
from evaluation.utilities import evaluate_selection, generate_roc_data, generate_pr_auc
from pathlib import Path
import json

def evaluate_model(
  configuration,
  path: Path,
  debug: bool = False, 
):
  '''
  Docstring for test_model()
  Evaluate an entire test dataset using the trained model and return extended results. 
  '''
  
  # Create Output Paths
  Path('../../metrics/datapoints').mkdir(parents = True, exist_ok = True)
  Path('../../metrics/figures').mkdir(parents = True, exist_ok = True)
    
  # Load Model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = reload_model(configuration, path)
  model = model.to(device)
  
  # Load Dataset. 
  _, test_loader = prepare_loaders(configuration)
  
  # Get a random sample from the test dataset.
  # random_selection = random.randint(0, len(test_loader))
  random_selection = 0
  
  # Store Predictions and Truths
  TRUTHS = []
  PREDICTIONS = []
  
  # Initialize Progress Bar
  progress_bar = enumerate(tqdm(test_loader,
            total = len(test_loader),
            desc = ''.join(['[INFO] Testing Model']),
            dynamic_ncols = True,
            leave = False,
            unit = 'image'
  ))
  
  jaccard_single = None
  dice_single = None
  
  # Start Validation Loop
  try:
    
    for n, (images, masks) in progress_bar:   
      image = images.to(device, dtype=torch.float)
      mask = masks.to(device, dtype=torch.float)
      
      with torch.no_grad():
        prediction = model(image)
        
      if n == random_selection:
        jaccard_single, dice_single = evaluate_selection(image, mask, prediction)
      
      PREDICTIONS.append(torch.nn.Sigmoid()(prediction))
      TRUTHS.append(mask)
      
      if debug: 
        break
          
  except KeyboardInterrupt:
    pass
    

  TRUTHS = torch.cat(TRUTHS, dim=0).long()
  ESTIMATIONS = torch.cat(PREDICTIONS, dim = 0)
  PREDICTIONS = (torch.cat(PREDICTIONS, dim=0) > 0.5).long()
  
  # For Dataset Summary Metrics
  jaccard = 1. - JaccardLoss(mode = 'binary', from_logits = False)(PREDICTIONS, TRUTHS).detach().cpu().numpy()
  dice = 1. - DiceLoss(mode = 'binary', from_logits = False)(PREDICTIONS, TRUTHS).detach().cpu().numpy()
  
  # For Dataset ROC, PR-Curve, AUC
  ground_truth = TRUTHS.cpu().numpy().flatten()
  estimations = ESTIMATIONS.cpu().numpy().flatten()
  
  # Output ROC JSON
  auc_value = generate_roc_data(
    Path('../metrics/datapoints'),
    ground_truth,
    estimations
  )
  
  # Output PR-Curve JSON and PR-AUC
  prauc_value = generate_pr_auc(
    Path('../metrics/datapoints'),
    ground_truth,
    estimations
  )
      
  # Generate Structured JSON for all metrics reported
  summary_file = Path('../metrics/datapoints') / 'summary.json'
  with open(summary_file, 'w') as summary:
    json.dump(
      {
        "evaluate": {
          "random_selection": {
            "IoU": jaccard_single,
            "Dice": dice_single, 
          },
          "dataset": {
            "IoU": jaccard,
            "Dice": dice,
            "ROC_AUC": auc_value,
            "PR_AUC": prauc_value
          }
        }
      },
      summary,
      indent = 4
    )
  
  LOGGER.info('Finished Model Evaluation.')
  