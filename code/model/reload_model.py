from pathlib import Path
from model.create_model import create_model
import torch


def reload_model(
  configuration,
  path: Path,
):
  '''
  Docstring for reload_model()
  '''
  model = create_model(configuration)
  model.load_state_dict(torch.load(path, map_location=torch.device(configuration.device)))
  model.eval()
  return model