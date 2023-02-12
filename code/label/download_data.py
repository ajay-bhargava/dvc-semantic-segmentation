import torch, torchvision
from pathlib import Path


def download_data() -> None:
    '''
    Uses Torchvision to download data to the Dataset Directory
    '''
    path = Path('../../../Datasets/').resolve()
    path.mkdir(exist_ok = True)
    torchvision.datasets.OxfordIIITPet(path, download = True)
    print('Done!')

if __name__ == '__main__':
    download_data()
