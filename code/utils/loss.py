import segmentation_models_pytorch.losses as losses
from log.logger import LOGGER

class Criterion:
  def __init__(self, configuration):
    '''
    Docstring for Loss()
    Make a choice of what Loss function to use in the model based on the configuration file.
    '''
    
    self.configuration = configuration    
    if self.configuration.retrieve('train.loss.Jaccard') is not None:
      self.loss = losses.JaccardLoss(
        mode = self.configuration.retrieve('train.loss.Jaccard.mode'),
        from_logits = self.configuration.retrieve('train.loss.Jaccard.from_logits'),
      )

    else:
      LOGGER.error("Loss function not found. Please check the configuration file.")
      raise Exception()  
    
  def __call__(self, prediction, ground_truth):
    '''
    Docstring for __call__()
    Performs a loss calculation for a given loss function defined in the instantiation of the class. 
    '''
    return self.loss(prediction, ground_truth)