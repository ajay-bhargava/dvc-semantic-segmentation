from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tempfile import NamedTemporaryFile
from label_studio_converter import brush
import json

# Get file lists
trimasks = sorted([x.resolve() for x in Path('../../../Datasets/oxford-iiit-pet/annotations/trimaps').glob('*.png') if not any(x.name.startswith(y) for y in ['._', '._.'])])
images = sorted([x.resolve() for x in Path('../../../Datasets/oxford-iiit-pet/images/').glob('*.jpg')])

# Debug 
print(f'Number of trimaps: {len(trimasks)}')
print(f'Number of images: {len(images)}')

# Create output directory
Path(Path.home() / 'Desktop' / 'annotations').mkdir(parents=True, exist_ok=True)
Path(Path.home() / 'Desktop' / 'images').mkdir(parents=True, exist_ok=True)

# Helper functions
def pre_process_trimap(trimap) -> np.ndarray:
    '''
    Docstring for pre_process_trimap():
    Pre-processes the trimap to be used for training. This is specific to the OxfordIIITPet dataset.
    Input: np.ndarray
    Output: np.ndarray
    '''
    map = trimap.copy()
    map = map.astype(np.float32)
    map[map == 2.0] = 0.0
    map[(map == 1.0) | (map == 3.0)] = 1.0
    return map.astype(np.uint8)

# Main Function
def produce_label_studio_annotations(trimask_path, image_path) -> None:
    '''
    Docstring for produce_label_studio_annotations():
    Produces the dataset annotations for Labelstudio in the manner appropriate for the software. 

    Inputs: 
        trimask_path: Path to a trimap (pathlib.Path)
        image_path: Path to an image (pathlib.Path)
    '''
    # Load trimap and image
    with Image.open(image_path) as img, Image.open(trimask_path) as map:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        trimap = np.array(map)

        with NamedTemporaryFile(suffix = '.png') as file:
            processed_trimap = pre_process_trimap(trimap)
            Image.fromarray(processed_trimap * 255).save(file.name)
            annotation = brush.image2annotation(
                path = file.name,
                label_name = 'Pets',
                from_name = 'tag', 
                to_name = 'image',
                ground_truth = True, 
            )
            with open(Path.home() / 'Desktop' / 'annotations' /trimask_path.name.replace('.png', '.json'), 'w') as f:
                task = {
                    'data': {'image': 's3://dvc-datasets/semantic-segmentation/ingest/pet-dataset/images/' + image_path.name}, 
                    'annotations': [
                            annotation
                        ]
                }
                json.dump(task, f)

        img.save(Path.home() / 'Desktop' / 'images' /  image_path.name)

if __name__ == '__main__':
    for trimask, image in tqdm(zip(trimasks, images), total = len(trimasks)):
        produce_label_studio_annotations(trimask, image)