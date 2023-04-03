from loaders.dataset import IIITPetDataset
from torch.utils.data import DataLoader

def prepare_loaders(configuration, fold: int = 1):
  '''
  Prepare DataLoaders for Test and Train DataSets
  '''
  # Create Datasets
  training_dataset = IIITPetDataset(dataset_path = configuration.dataset_path, mode = 'train', longest_dimension = configuration.retrieve('train.hyperparameters.image_longest_dimension'))
  testing_dataset = IIITPetDataset(dataset_path = configuration.dataset_path, mode = 'test', longest_dimension = configuration.retrieve('train.hyperparameters.image_longest_dimension'))

  # Create DataLoader(s)
  training_loader = DataLoader(
    training_dataset, 
    batch_size = configuration.retrieve('train.hyperparameters.batch_size'),
    num_workers = configuration.retrieve('train.hyperparameters.num_workers')
  )

  testing_loader = DataLoader(
    testing_dataset, 
    batch_size = configuration.retrieve('test.hyperparameters.batch_size'),
    num_workers = configuration.retrieve('test.hyperparameters.num_workers')
  )

  return training_loader, testing_loader
