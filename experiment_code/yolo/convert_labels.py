import os
import json
import cv2

#### convert Kaggle CoCo format(json) to Yolov5 Format(txt)

def convert_coco_bbox_to_yolo(coco_bbox, img_size):
    """
    Convert COCO bbox format to YOLOv5 format.

    Args:
    coco_bbox (list): a list of four numbers representing the COCO bbox, [x, y, width, height].
    img_size (tuple): a tuple of two numbers representing the size of the image, (img_width, img_height).

    Returns:
    yolo_bbox (list): a list of four numbers representing the YOLOv5 bbox, [x_center, y_center, width, height].
    """
    x, y, width, height = coco_bbox
    img_width, img_height = img_size

    # Convert COCO bbox (top left x, top left y, width, height) to YOLO format (center x, center y, width, height)
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height
    width = width / img_width
    height = height / img_height

    yolo_bbox = [x_center, y_center, width, height]

    return yolo_bbox


labels = json.load(open('./yolov5/datasets/train_annotations'))
for i in labels:
    image_id = i['image_id']
    bbox = i['bbox']
    category_id = str(i['category_id'] - 1)

    image_name = 'image_id_' + str(image_id).zfill(3) + '.jpg'
    image_path = os.path.join('./yolov5/datasets/images/train', image_name)
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    yolo_bbox = convert_coco_bbox_to_yolo(bbox, (width, height))

    string = category_id + ' ' + ' '.join([str(a) for a in yolo_bbox]) + '\n'
    txt_name = image_name.replace('.jpg', '.txt')
    with open(os.path.join('./yolov5/datasets/images/train', txt_name), 'w') as f:
        f.write(string)
        print("Convert {}".format(txt_name))
        f.close()

labels = json.load(open('./yolov5/datasets/valid_annotations'))
for i in labels:
    image_id = i['image_id']
    bbox = i['bbox']
    category_id = str(i['category_id'] - 1)

    image_name = 'image_id_' + str(image_id).zfill(3) + '.jpg'
    image_path = os.path.join('./yolov5/datasets/images/valid', image_name)
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    yolo_bbox = convert_coco_bbox_to_yolo(bbox, (width, height))

    string = category_id + ' ' + ' '.join([str(a) for a in yolo_bbox]) + '\n'
    txt_name = image_name.replace('.jpg', '.txt')
    with open(os.path.join('./yolov5/datasets/images/valid', txt_name), 'w') as f:
        f.write(string)
        print("Convert {}".format(txt_name))
        f.close()
