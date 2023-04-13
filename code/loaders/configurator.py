import yaml
from functools import reduce

class CONFIGURATOR(object):
  '''
  A class that contains all the configuration options for the training process. 
  We are going to use this in place of a YAML file so that parameters are easily visible in the code as dot strings. 
  '''
  def __init__(self, yaml_path, dataset_path, device):
    self.path = yaml_path
    self.dataset_path = dataset_path
    self.device = device
    with open(self.path, 'r') as f:
      self.dictionary = yaml.safe_load(f)
    self.class_map = self.retrieve('test.hyperparameters.classes')
    
  def retrieve(self, keys, default=None):
      return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), self.dictionary)
    
  def get_classes(self) -> list:
    return [{"name": x, "id": y} for x, y in self.class_map.items()]
  
  def get_class_labels(self):
    return {value: key for key, value in self.class_map.items()}