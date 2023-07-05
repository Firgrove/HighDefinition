import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torchvision
import json
import os
from torchvision.io import read_image
import matplotlib.pyplot as plt
from random import randrange
import cv2

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_path, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(image_dir))))
        with open(annotation_path) as json_file:
            annotation_list = json.load(json_file)
        self.annotation_list = annotation_list

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.imgs[idx])
        label_bbox = self.annotation_list[idx]['bbox']
        img = read_image(img_path).numpy().transpose(1, 2, 0) # Convert CHW to HWC format for albumentations
        # Pascal VOC / original annotation is COCO
        boxes = [[label_bbox[0], label_bbox[1], label_bbox[0]+label_bbox[2], label_bbox[1]+label_bbox[3]]]  # In a list for albumentations
        category = [self.annotation_list[idx]['category_id']]  # In a list for albumentations

        if self.transforms:
            transformed = self.transforms(image=img, bboxes=boxes, labels=category)
            img = transformed["image"]
            boxes = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            category = torch.as_tensor(transformed["labels"], dtype=torch.int8)

        image_id = torch.tensor([idx])
        target = {}
        target["boxes"] = boxes
        target["category"] = category
        target["image_id"] = image_id
        
        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transforms(): # add normalization later
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.25),
            A.Rotate(limit=25, p=0.75),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


if __name__ == "__main__":
    train_dataset = CustomImageDataset('./images/train', annotation_path="./images/train_annotations", transforms=get_transforms())
    # Retrieve the first image in the Dataset
    print(train_dataset[0][0])
    # Retrieve the label for first image
    print("-----------------------")
    print(train_dataset[0][1])
    
    