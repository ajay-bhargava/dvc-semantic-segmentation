import yaml
from functools import reduce

class CONFIGURATOR(object):
  '''
  A class that contains all the configuration options for the training process. 
  We are going to use this in place of a YAML file so that parameters are easily visible in the code as dot strings. 
  '''
  def __init__(self, path):
    self.path = path
    with open(self.path, 'r') as f:
      self.dictionary = yaml.safe_load(f)

  def retrieve(self, keys, default=None):
      return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), self.dictionary)
