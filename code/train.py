# Torch Imports
import torch

# Python Imports
from pathlib import Path
from argparse import ArgumentParser, Namespace

# Miscellaneous Libraries
from log.logger import LOGGER
from loaders.configurator import CONFIGURATOR

def parse_command_options() -> Namespace:
  '''
  Creates an ArgumentParser object that contains all the arguments for running the script.
  '''
  parser = ArgumentParser(description='Train the UNet on images and target masks')
  parser.add_argument('--config', '-c', metavar='C', type=str, default='train.yaml', help='Path to the configuration file')
  parser.add_argument('--dataset', '-d', metavar='D', type=str, default='data/computed/pets-dataset/train', help='Path to the dataset')
  arguments = parser.parse_args()
  return arguments

def train_model(
  dataset_path: Path,
) -> None:
  '''
  Performs the act of training an UNet model. 
  '''


if __name__ == "__main__":
  # Parse Arguments
  arguments = parse_command_options()

  # Set the device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' if not torch.backends.mps.is_available() else 'mps')  # type: ignore (this is a MacOS specific setting)

  # Log the device
  LOGGER.info("🤖 Welcome to UNet.")
  LOGGER.info("Using {}".format(str(device).upper()))

  # Load the configuration file
  CONFIG = CONFIGURATOR(arguments.config, arguments.dataset, device)
  