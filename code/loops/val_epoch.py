import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loss import Criterion
from log.logger import LOGGER
import wandb, math

from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss

def _validate_an_epoch(
  configuration,
  wandb_table: wandb.Table,
  model: torch.nn.Module,
  dataloader: DataLoader,
  device, 
  epoch: int,
  test: bool = False,
):
  '''
  Docstring for _validate_an_epoch()
  Perform validation of a model following an epoch of training. 
  '''
  model.eval()
  
  # Static Variables (Per Epoch)
  dataset_size = 0
  running_loss = 0.0

  # Initialize Loss Function
  criterion = Criterion(configuration)
  
  # Validation Metrics Storage
  TRUTHS = []
  PREDICTIONS = []
  VALIDATION_SCORES = []
  EPOCH_LOSS = 0
      
  
  # Initialize wandb.Classes()
  class_set = wandb.Classes(
    configuration.get_classes()
  )
  
  # Loop
  with tqdm(dataloader, 
          total=len(dataloader), 
          desc = ''.join(['Validation Epoch ', str(epoch)]), 
          dynamic_ncols = True, 
          leave = False) as pbar:
    try:
      
      # Step counter
      STEP_COUNTER = 0

      for images, masks in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)
        
        # Batch Size
        batch_size = images.shape[0]
        
        # Run a prediction
        prediction = model(images)
        loss = criterion(prediction, masks)
        
        # Update Dataset Iterated
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        # Calculate epoch loss
        EPOCH_LOSS = running_loss / dataset_size
        
        # Update memory usage
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)

        # Update the progress bar
        pbar.set_postfix(
          gpu_memory = mem,
          loss = f'{EPOCH_LOSS:0.4f}',
        )
        
        # Perform a nn.Sigmoid() on the prediction to convert to a probability
        PREDICTIONS.append(torch.nn.Sigmoid()(prediction))
        TRUTHS.append(masks)
        STEP_COUNTER += batch_size
        
        if (STEP_COUNTER % math.floor(len(dataloader) / 10)) == 0 or test:          
          for x in range(0, prediction.shape[0]):
            prediction_logging = wandb.Image(
              data_or_path = images[x,...].permute(1,2,0).detach().cpu().numpy(), 
              masks = {
                "predictions": {
                  "mask_data": (prediction[x,...] > 0.5).long().squeeze().detach().cpu().numpy(), 
                  "class_label": configuration.get_class_labels()
                  },
                
                "ground_truth": {
                  "mask_data": masks[x,...].long().detach().cpu().numpy(),
                  "class_label": configuration.get_class_labels()
                }
              },
              classes = class_set,
            )
            wandb_table.add_data(prediction_logging)
          LOGGER.info('Logging validation pair to wandb.Table ✨')
          
        if test:
          break

      # Retain prediction with probability of > 0.5
      TRUTHS = torch.cat(TRUTHS, dim=0).long()
      PREDICTIONS = (torch.cat(PREDICTIONS, dim=0) > 0.5).long()
      
      # Calculate Loss for this Validation Epoch
      jaccard = 1. - JaccardLoss(mode = 'binary', from_logits = False)(PREDICTIONS, TRUTHS).detach().cpu().numpy()
      dice = 1. - DiceLoss(mode = 'binary', from_logits = False)(PREDICTIONS, TRUTHS).detach().cpu().numpy()
      VALIDATION_SCORES.append({"Dice": dice, "Jaccard": jaccard})

      return EPOCH_LOSS, VALIDATION_SCORES
        
    except KeyboardInterrupt:
      LOGGER.error('Keyboard Interrupt. Exiting...')
      return EPOCH_LOSS, [{"Dice": None, "Jaccard": None}]
    
