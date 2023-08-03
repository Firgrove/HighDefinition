# Yolov5 Experiments

1. Please see experiment record from [experiment_record.ipynb](./experiment_record.ipynb).
2. Experiment results are stored in the folder `experiment`.
3. [convert_labels.py](./convert_labels.py) is for converting labels.
4. [getResult.py](./getResult.py) is for evaluating results.

For evaluation, you can download our yolo models from this link [https://drive.google.com/drive/folders/1uOf7ET\_-WAHoYVULlqnLyd6LPt9LFU60?usp=drive_link](https://drive.google.com/drive/folders/1uOf7ET_-WAHoYVULlqnLyd6LPt9LFU60?usp=drive_link), then modify the `--weights` flag in the notebook to get the results.


## Quick Start for Detection and Evaluation
1. `pip install -r requirements.txt`
2. Download the weight from [https://drive.google.com/drive/folders/1uOf7ET\_-WAHoYVULlqnLyd6LPt9LFU60?usp=drive_link](https://drive.google.com/drive/folders/1uOf7ET_-WAHoYVULlqnLyd6LPt9LFU60?usp=drive_link), put the weight file `pretrained.pt` in the yolov5 root folder.
3. Put images and labels as this diretory format
    - yolov5
      - datasets
           - train
               - image_id_000.jpg
               - ...
           - valid
               - image_id_000.jpg
               - ...
           - train_annotations
           - valid_annotations

4. `python3 detect.py --source './datasets/valid' --weights 'pretrained.pt' --project 'evaluation' --max-det 1 --save-txt` 
5. See prediction result(output images) in folder `yolov5/evaluation/exp`
6. `python3 convert_labels.py`
7. `python3 getResult.py --predict_label_folder './evaluation/exp/labels'`