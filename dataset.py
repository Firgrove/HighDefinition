import numpy as np
import torch
import torchvision
import json
from torchvision.io import read_image
import os

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
        img = read_image(img_path).to(torch.float32)
        # Pascal VOC / original annotation is COCO
        boxes = [label_bbox[0], label_bbox[1], label_bbox[0]+label_bbox[2], label_bbox[1]+label_bbox[3]]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        category = torch.as_tensor((self.annotation_list[idx]['category_id'],), dtype=torch.int8)
        image_id = torch.tensor([idx])
        target = {}
        target["boxes"] = boxes
        target["category"] = category
        target["image_id"] = image_id
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    train_dataset = CustomImageDataset('./images/train', annotation_path="./images/train_annotations", transforms=None)

    # Retrieve the first image in the Dataset
    print(train_dataset[0][0])
    # Retrieve the label for first image
    print("-----------------------")
    print(train_dataset[0][1])