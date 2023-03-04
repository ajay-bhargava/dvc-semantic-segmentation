from torch.utils.data import Dataset
from pathlib import Path

class IIITPetDataset(Dataset):
        def __init__(self, dataset_path: Path, img_scale: float = 0.5):
                self.images_dir = Path(dataset_path)