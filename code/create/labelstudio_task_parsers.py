import dateutil.parser as DateParser
from label_studio_converter.brush import decode_rle as Decode
import json, numpy as np 

def Split_S3(s3_path):
    path_parts=s3_path.replace("s3://","").split("/")
    bucket=path_parts.pop(0)
    key="/".join(path_parts)
    return bucket, key

class LabelStudioDataSet:
    def __init__(self, path) -> None:
        with open(path) as f:
            self.dataset = json.load(f)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]

    def __iter__(self):
        for item in self.dataset:
            yield item

class LabelStudioData:
    def __init__(self, task) -> None:
        self.task = task
        self.latest_annotation = self.resolve_latest_annotation()
        self.task_details = self.get_task_details()
        
    def __len__(self):
        return len(self.task['annotations'])
    
    def resolve_latest_annotation(self) -> dict:
        '''
        Resolve the latest annotation if there is more than one. 
        '''
        if len(self) == 1:
            return self.task['annotations'][0]
        else:
            return sorted(self.task['annotations'], key=lambda x: DateParser.parse(x['created_at']))[-1]
        
    def get_image_url(self) -> tuple:
        return Split_S3(self.task['data']['image'])[0], Split_S3(self.task['data']['image'])[1]
    
    def get_task_details(self) -> list:
        '''
        Return all the np.ndarrays associated with the RLE's present in the task. 
        '''
        output = []
        counter = 0
        for result in self.latest_annotation['result']:
            if result['type'] == 'brushlabels': 
                #TODO: Add support for other types, and #WARNING: This is a hack.
                image = Decode(result['value']['rle'])
                width = result['original_width']
                height = result['original_height']
                counter += 1 
                label = result['value']['brushlabels'][0]
                image_output = np.reshape(image, [height, width, 4])[...,3]
                output.append({'array': image_output, 'label': label, 'uri': self.get_image_url()})
        return output