# Torch Imports
import torch
from torch.utils.data import DataLoader

# Model Libraries
import segmentation_models_pytorch as smp

# Python Imports
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

# Training Libraries
from loaders.dataset import IIITPetDataset

# Miscellaneous Libraries
import logging
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
  # 1. Create the dataset and dataloader
  try:
    dataset = IIITPetDataset(dataset_path)
  except (AssertionError, RuntimeError, IndexError):
    print('Error: The dataset path is invalid. Please check the path and try again.')
    return

if __name__ == "__main__":
  # Parse Arguments
  arguments = parse_command_options()

  # Configure Logging
  logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info("🤖 Welcome to UNet.")
  logging.info("Using {}".format(str(device).upper()))

  # Load the configuration file
  CONFIG = CONFIGURATOR(arguments.config)

  # Train the model
