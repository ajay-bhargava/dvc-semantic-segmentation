# Torch Imports
import torch, os
import wandb

# Python Imports
from argparse import ArgumentParser, Namespace

# Created Libraries
from log.logger import LOGGER
from loaders.configurator import CONFIGURATOR
from loops.run import run_loops
from model.create_model import create_model

def parse_command_options() -> Namespace:
  '''
  Creates an ArgumentParser object that contains all the arguments for running the script.
  '''
  parser = ArgumentParser(description='Train the UNet on images and target masks')
  parser.add_argument('--config', '-c', metavar='C', type=str, default='control.yaml', help='Path to the configuration file')
  parser.add_argument('--dataset', '-d', metavar='D', type=str, default='data/computed/pets-dataset', help='Path to the dataset')
  arguments = parser.parse_args()
  return arguments

def train(
  configuration,
) -> None:
  '''
  Performs the act of training an UNet model. 
  '''
  run = wandb.init(
    project = ''.format(configuration.retrieve('train.project_name')),
    config = configuration.dictionary,
    mode = 'disabled',
    anonymous = 'must'
  )
  model = create_model(configuration)
  optimizer = torch.optim.Adam(model.parameters(), 
                               lr = configuration.retrieve('train.hyperparameters.learning_rate'))
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma = 0.25)
  trained_model = run_loops(configuration, 
                            optimizer, 
                            scheduler, 
                            model, 
                            run)
  
  run.finish() # type: ignore

if __name__ == "__main__":
  # Parse Arguments
  arguments = parse_command_options()

  # Set the device
  device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore (this is a MacOS specific setting)

  # Log the device
  LOGGER.info("🤖 Welcome to UNet.")
  if device != "cuda":
    LOGGER.warning("🔥 No GPU detected. Training will be slow.")
  else:
    LOGGER.info("🔥 GPU detected. Training using {}.".format(torch.cuda.get_device_name(torch.device("cuda"))))

  # Load the configuration file
  configuration = CONFIGURATOR(arguments.config, arguments.dataset, device)
  
  # Train the model
  train(configuration)
  if os.path.exists('../wandb/'):
    os.remove('../wandb/')
  LOGGER.info("🎉 Exiting.")
  