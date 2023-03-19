from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from skimage import transform
from log.logger import LOGGER
import cv2

class Albumentations:
  '''
  Class Albumentations()
  This class is used to create a wrapper around the albumentations library for semantic segmentation . This is a model hyperparameter attribute. 
  '''
  def __init__(self, mode: str = "train", longest_dimension: int = 512):
    self.transform = None
    try:
      import albumentations as A
      from albumentations.pytorch import ToTensorV2

      if mode == "train":
      
        T = [
          A.LongestMaxSize(max_size = longest_dimension, always_apply = True),
          A.PadIfNeeded(min_height = longest_dimension, min_width = longest_dimension, border_mode = cv2.BORDER_CONSTANT, value = [0,0,0], mask_value= [0, 0, 0], always_apply = True),
          A.ShiftScaleRotate(
            shift_limit = 0.2,
            scale_limit = 0.2,
            rotate_limit = 20,
            p = 0.5
          ),
          A.RandomBrightnessContrast(
            brightness_limit = 0.3,
            contrast_limit = 0.3,
            p = 0.5
          ),
          A.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
          ),
          ToTensorV2()
        ]
        self.transform = A.Compose(T)
      
      elif mode == "test":
      
        T = [
          A.LongestMaxSize(max_size = longest_dimension, always_apply = True),
          A.PadIfNeeded(min_height = longest_dimension, min_width = longest_dimension, border_mode = cv2.BORDER_CONSTANT, value = [0,0,0], mask_value= [0, 0, 0], always_apply = True),
          A.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
          ),
          ToTensorV2()
        ]
        self.transform = A.Compose(T)

    except Exception as e:
      LOGGER.info("Albumentations not installed. Please install it using pip install albumentation.")

  def __call__(self, image, mask, p = 1.0):
    if self.transform and np.random.rand() < p:
      new = self.transform(image = image, mask = mask)
      image, mask = new['image'], new['mask']
    return image, mask

class IIITPetDataset(Dataset):
  def __init__(self, dataset_path: Path, mode: str = "test", longest_dimension: int = 512):
    '''
    Class IIITPetDataset()
    DataSet class for the IIIT Pet Dataset.
    '''
    self.images_dir = Path(dataset_path)
    self.albumentations = Albumentations(mode = mode, longest_dimension = longest_dimension)
    if mode == "train":
      self.images_dir = self.images_dir / "train"
      self.paths = [x.resolve() for x in self.images_dir.glob("*.jpg")]
    elif mode == "test":
      self.images_dir = self.images_dir / "test"
      self.paths = [x.resolve() for x in self.images_dir.glob("*.jpg")]
    else:
      LOGGER.error("Incorrect Mode. Mode should be either 'train' or 'test'.")

  def __len__(self):
    return len(self.paths)
  
  def __getitem__(self, index):

    impath, numpath = self.paths[index], self.paths[index].with_suffix('.npy')
    image, mask = np.asarray(Image.open(impath)), np.load(numpath)
    
    # Call Transformations
    image, mask = self.albumentations(image, mask)
    return image, mask