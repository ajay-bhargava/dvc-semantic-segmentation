import logging, os

def set_logger(name = None):
  '''
  Set the logging function and call it. 
  '''
  log = logging.getLogger(name)
  log.setLevel(logging.INFO)
  handler = logging.StreamHandler()
  handler.setLevel(logging.INFO)
  handler.setFormatter(logging.Formatter('\n [%(levelname)s] %(message)s'))
  log.addHandler(handler)

set_logger()
LOGGER = logging.getLogger("UNet")
