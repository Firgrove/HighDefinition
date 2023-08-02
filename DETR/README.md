# HighDefinition - DETR (DEtection Transformer)

To train and test the DETR algorithm, run the following commands from the DETR directory.

```!pip3 install requirements.txt

You can now either train the model yourself or download our pretrained model from [google drive](https://drive.google.com/file/d/1TB7vYB7PmeyE_ryDhrczXNZujgK-4uKG/view?usp=drive_link).

### Training

To train the model run the following command:

```!python train.py -m face_detr -b 8 -e 500 --cuda

This will train the model with some sensible presets. More details can be found in the [DETR_eval.ipynb](DETR_eval.ipynb) notebook. The train function will save the model at the end of each epoch, if the model has improved.

### Evaluating Results

To produce metrics for the trained model go to [DETR_eval.ipynb](DETR_eval.ipynb). This notebook contains code to visualise and evaluate the model.