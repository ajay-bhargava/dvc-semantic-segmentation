import segmentation_models_pytorch as smp
from log.logger import LOGGER

def create_model(configuration):
    '''
    Docstring for create_model()
    Uses the SMP Pytorch framework to configure a model. 
    '''
    model = smp.Unet(
        encoder_name = configuration.retrieve('train.hyperparameters.encoder_name'),
        encoder_weights = configuration.retrieve('train.hyperparameters.encoder_weights'),
        in_channels = configuration.retrieve('train.hyperparameters.in_channels'),
        classes = configuration.retrieve('train.hyperparameters.classes'),
    )
    model = model.to(configuration.device)
    LOGGER.info('🤖 is ready.')
    return model