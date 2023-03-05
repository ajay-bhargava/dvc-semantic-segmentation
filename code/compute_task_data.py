from pathlib import Path
import numpy as np, boto3
from create.labelstudio_task_parsers import LabelStudioData, LabelStudioDataSet
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from tqdm import tqdm

S3 = boto3.client('s3')

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Input LabelStudio Dataset Path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    return parser

def compute_task_data() -> None:
    '''
    Performs the derivation of test and train numpy arrays from the LabelStudio dataset
    '''
    # Parse the arguments
    ARGUMENTS = argument_parser().parse_args()

    # Load the dataset and split it into train and test
    dataset = LabelStudioDataSet(Path(ARGUMENTS.dataset).resolve())
    
    # Split the Dataset
    train, test = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)

    # Build the output test and train directory
    output_root = Path(ARGUMENTS.output).resolve()
    Path(output_root / 'train').mkdir(parents=True, exist_ok=True)
    Path(output_root / 'test').mkdir(parents=True, exist_ok=True)
    
    # Iterate through the data. Save the image and label. 
    for dataset, name in zip([train, test], ['train', 'test']):
        counter = 0
        pbar = tqdm(dataset, desc='Processing {} data.'.format(name), total=len(dataset), unit = 'images', leave=False)
        for data in pbar:
            # Derive the image and label
            datum = LabelStudioData(data).task_details
            for element in datum:
                counter += 1
                # Get key value pairs from structured dictionary
                mask = element['array']
                uri = element['uri']

                # Save the image and label
                np.save(Path(output_root / name / '{}.npy'.format(str(counter).zfill(4))), mask)
                jpeg_path = str(Path(output_root / name / '{}.jpg'.format(str(counter).zfill(4))).resolve())
                S3.download_file(uri[0], uri[1], jpeg_path)
            
if __name__ == '__main__':
    compute_task_data()
    print('Finished computing data.')