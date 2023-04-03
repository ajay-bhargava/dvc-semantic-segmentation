import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loss import Criterion
from log.logger import LOGGER

def _do_an_epoch(
        configuration, 
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        dataloader: DataLoader,
        device,
        epoch: int,
) -> float:
  '''
  Docstring for _do_an_epoch()
  '''
  model.train()
  
  # Enable mixed precision 
  scaler = torch.cuda.amp.GradScaler(enabled = configuration.retrieve('train.hyperparameters.mixed_precision'))
  
  # Static Variables (Per Epoch)
  dataset_size = 0
  running_loss = 0.0
  epoch_loss = 0.0
  
  # Initialize loss function
  criterion = Criterion(configuration)
  
  # Training Loop
  with tqdm(dataloader, 
            total=len(dataloader), 
            desc = ''.join(['Epoch ', str(epoch)]), 
            dynamic_ncols = True, 
            leave = False) as pbar:
    try:
      
      for images, masks in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)
        
        # Batch Size
        batch_size = images.shape[0]

        # Forward Pass (Mixed Precision - FP16 or Lower)
        with torch.autocast(device_type = device):
          prediction = model(images)
          loss = criterion(prediction, masks)
          loss = loss / configuration.retrieve('train.hyperparameters.loss_smoothing') # Scaling will affect the trade-off between prediction loss and regularization
          
        # BackPropagation 
        scaler.scale(loss).backward() # type: ignore
        
        if dataset_size % configuration.retrieve('train.hyperparameters.loss_smoothing') == 0:
          # Step the optimizer and update the learning rate in a more smoothed way. 
          scaler.step(optimizer)
          scaler.update()
        
          optimizer.zero_grad()
          
          # Update the learning rate based on the scheduler. 
          if scheduler is not None:
            scheduler.step()
          
        # Log
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        # Calculate epoch loss
        epoch_loss = running_loss / dataset_size
        
        # Update the progress bar
        pbar.set_postfix(
          gpu_memory = mem,
          lr = optimizer.param_groups[0]['lr'],
          loss = f'{epoch_loss:0.4f}',
        )
        torch.cuda.empty_cache()
      
      return epoch_loss
    
    except KeyboardInterrupt:
      torch.cuda.empty_cache()
      LOGGER.error('Keyboard Interrupt. Exiting...')
      return epoch_loss
      
    

      
    
    
      

      
 