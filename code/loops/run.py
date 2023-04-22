import torch, wandb, time, copy, numpy as np
from pathlib import Path
from log.logger import LOGGER
from loops.train_epoch import _train_an_epoch
from loops.val_epoch import _validate_an_epoch
from loaders.prepare import prepare_loaders

def run_loops(
  configuration,
  optimizer: torch.optim.Optimizer,
  scheduler: torch.optim.lr_scheduler._LRScheduler,
  model: torch.nn.Module,
  run,
):
  '''
  Docstring for run_training_loops()
  '''
  # ✨ Weight & Biases Tracking
  wandb.watch(models = model, log_freq = 10, log_graph = True)
  table = wandb.Table(columns = ['ground_truth', 'predictions'])
  
  # ⏱️ Metrics Tracking Start
  start = time.time()
  best_weights = copy.deepcopy(model.state_dict())
  best_dice_score = -np.inf
  best_epoch = -1
  
  # Define Early Stopping
  patience = configuration.retrieve('train.hyperparameters.patience')
  delta = configuration.retrieve('train.hyperparameters.delta')
  counter = 0
  
  # Load the Dataset 🪨
  train_loader, val_loader = prepare_loaders(configuration)
  
  # Create Folder for Saving Models
  Path('./models').mkdir(parents = True, exist_ok = True)
  
  # Start the loop
  try:
    for epoch in range(1, configuration.retrieve('train.hyperparameters.epochs') + 1):
      LOGGER.info(''.join(['Epoch ', str(epoch), ' of ', str(configuration.retrieve('train.hyperparameters.epochs'))]))
      
      # Start the ∞ training loop
      train_loss = _train_an_epoch(
        configuration = configuration,
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        dataloader = train_loader,
        device = configuration.device,
        epoch = epoch
      )
      
      # ∞ Start the validation loop
      validation_loss, validation_scores = _validate_an_epoch(
        configuration = configuration,
        wandb_table = table, 
        model = model, 
        dataloader = val_loader,
        device = configuration.device,
        epoch = epoch
      )
      
      dice, jaccard = validation_scores.values()
      
      # Submit Logging to Weight & Biases ✨
      wandb.log(
        data = {
          'train/loss': train_loss,
          'val/loss': validation_loss,
          'val/dice': dice,
          'val/jaccard': jaccard,
          'val/table': table,
        }, 
        step = epoch
      )
      
      LOGGER.info('Epoch {} of {} complete. Validation Loss: {:.3f} | Validation Dice: {:.3f} | Validation Jaccard: {:.3f}'.format(epoch, 
                                                                                                                                   configuration.retrieve('train.hyperparameters.epochs'), 
                                                                                                                                   validation_loss, 
                                                                                                                                   dice, 
                                                                                                                                   jaccard))
      
      # Deep copy and save the best model
      if dice >= best_dice_score:
        LOGGER.info('Validation dice score improved from {:.3f} to {:.3f}. Saving model.'.format(best_dice_score, dice))
        best_dice_score = dice
        best_jaccard_score = jaccard
        best_epoch = epoch
        run.summary['val/best_dice'] = best_dice_score
        run.summary['val/best_jaccard'] = best_jaccard_score
        run.summary['train/best_epoch'] = best_epoch
        best_weights = copy.deepcopy(model.state_dict())
        BEST_SAVE_PATH = './models/best_model.pth'
        torch.save(model.state_dict(), BEST_SAVE_PATH)
        LOGGER.info('Model Saved.')
        run.save(BEST_SAVE_PATH)
        
        
      # Early Stopping
      elif dice <= best_dice_score + delta:
        counter += 1
        if counter >= patience:
          LOGGER.info('Early Stopping.')
          break
        
      else:
        pass
      
      latest_epoch_weights = copy.deepcopy(model.state_dict())
      LATEST_SAVE_PATH = './models/latest_model.pth'
      torch.save(model.state_dict(), LATEST_SAVE_PATH)
      
    end = time.time()
    LOGGER.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format((end - start) // 3600, (end - start) % 3600 // 60, (end - start) % 60))
    LOGGER.info('Best Validation Dice: {:.3f} | Best Epoch: {}'.format(best_dice_score, best_epoch))
    
    return model
    
  except KeyboardInterrupt:
    LOGGER.info('Training interrupted. Saving model.')
    

  
  