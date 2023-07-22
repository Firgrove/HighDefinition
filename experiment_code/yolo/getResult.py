import os
import cv2
import json
import math
import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse

def compute_iou(x1, y1, w1, h1, x2, y2, w2, h2):
    # Calculate the (x, y)-coordinates of the intersection rectangle
    inter_top_left_x = max(x1, x2)
    inter_top_left_y = max(y1, y2)
    inter_bottom_right_x = min(x1 + w1, x2 + w2)
    inter_bottom_right_y = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, inter_bottom_right_x - inter_top_left_x) * max(0, inter_bottom_right_y - inter_top_left_y)

    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = inter_area / float(box1_area + box2_area - inter_area)

    # return the intersection over union value
    return iou

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def get_metrics(annotation_path, image_folder, predict_label_folder):
    labels = json.load(open(annotation_path))
    iou_list = []
    distance_list = []
    target_label = []
    predict_label = []

    for i in labels:
        image_id = i['image_id']
        bbox = i['bbox']
        category_id = str(i['category_id'] - 1)

        image_name = 'image_id_' + str(image_id).zfill(3) + '.jpg'
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        label_name = image_name.replace('.jpg', '.txt')
        predictLabel_path = os.path.join(predict_label_folder, label_name)
        targetLabel_path = os.path.join(image_folder, label_name)
        height, width, _ = image.shape

        with open(targetLabel_path) as target_file:
            target_content = target_file.read().split()
            target_category = target_content[0]
            target_centerX = float(target_content[1]) * width
            target_centerY = float(target_content[2]) * height
            target_width = float(target_content[3]) * width
            target_height = float(target_content[4]) * height
            target_label.append(target_category)

            #print(target_centerX,target_centerY,target_width,target_height)

        try:
            with open(predictLabel_path) as predict_file:
                predict_content = predict_file.read().split()
                predict_category = predict_content[0]
                predict_centerX = float(predict_content[1]) * width
                predict_centerY = float(predict_content[2]) * height
                predict_width = float(predict_content[3]) * width
                predict_height = float(predict_content[4]) * height
                predict_label.append(predict_category)
                #print(content)
                iou = compute_iou(target_centerX, target_centerY, target_width, target_height, predict_centerX, predict_centerY, predict_width, predict_height)
                iou_list.append(iou)

                distance = calculate_distance(target_centerX, target_centerY, predict_centerX, predict_centerY)
                distance_list.append(distance)

        except:
            #print(predictLabel_path)
            if target_category == '0':
                predict_label.append('1')
            else:
                predict_label.append('0')
    return iou_list, distance_list, target_label, predict_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This is a description of what my program does")
    parser.add_argument("--annotation_path", type=str, default='./datasets/valid_annotations' ,help="annotation_path")
    parser.add_argument("--image_folder", type=str, default='./datasets/images/valid', help="image_folder")
    parser.add_argument("--predict_label_folder", type=str, help="predict_label_folder")
    
    args = parser.parse_args()
    #print(args.num_epochs)
    target_names = ['Penguin', 'Turtle']
    
    iou_list, distance_list, target_label, predict_label = get_metrics(annotation_path=args.annotation_path, image_folder=args.image_folder, predict_label_folder=args.predict_label_folder)
    
    print(classification_report(target_label, predict_label, target_names=target_names))
    iou_list = np.array(iou_list)
    distance_list = np.array(distance_list)
    iou_std = np.std(iou_list)
    iou_mean = np.mean(iou_list)
    print('iou_std = {:.3f}, iou_mean = {:.3f}'.format(iou_std, iou_mean))
    distance_std = np.std(distance_list)
    distance_mean = np.mean(distance_list)
    print('distance_std = {:.3f}, distance_mean = {:.3f}'.format(distance_std, distance_mean))



        

    

