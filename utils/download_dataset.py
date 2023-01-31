import fiftyone
import shutil
from os import makedirs
import fiftyone.utils.splits as fous
dataset_path = "/home/ubuntu/Parth/object_counting/dataset/"

dataset_licence_plate = fiftyone.zoo.load_zoo_dataset(
    "coco-2017",
    classes=['person'],
    label_types=["detections"],
    dataset_name='dataset_person',
    dataset_dir='/home/ubuntu/Parth/object_counting/dataset/person',
    cleanup=True,
)

for t in ['train', 'test', 'validation']:
    dataset_licence_plate.match_tags(t).export(
        export_dir="/home/ubuntu/Parth/object_counting/dataset/YOLOv5/person/",
        dataset_type=fiftyone.types.YOLOv5Dataset,
        classes=['person'],
        split=t
    )