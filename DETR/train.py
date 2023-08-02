import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os

from torch.utils.data import DataLoader
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from dataset import CustomImageDataset
from custom_models import convNN, resnet18, resnet34, resnet50, DETR

import train_detr

CLASSES = ["turtle", "penguin"]

@torch.no_grad()
def evaluate(model, valid_loader):
    # Set model to eval mode
    # do eval
    # set model back to train mode
    # return loss
    pass

def train(model, train_loader, lr, device, valid_set, momentum=0.9, epochs=5):
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)#, momentum=momentum)

    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            images, labels = data   # images, age, gender, race, landmarks

            # Zero paramter gradients
            optimizer.zero_grad()
            #print(images.permute(0, 2, 3, 1).shape)
            images, labels = images.to(device), labels["boxes"].to(device)

            outputs = model(images)
            print(outputs[0].shape  )
            loss = loss_func(outputs[0], labels.view(-1, 4))
            loss.backward()
            optimizer.step()

            print(f"EpochL {epoch}, iteration: {i} had loss: {loss} and score: {evaluate(model, valid_set)}")
            
    return model

if __name__ == "__main__":
    # Read in arguments to specify which model to train and some parameters
    parser = ArgumentParser()
    parser.add_argument("-b", "--batch", 
                        help="Batch size for training.", 
                        type=int, 
                        metavar="INT",
                        default=64)
    parser.add_argument("-m", "--model",
                        help="Choose which model structure to use.",
                        default="face_detr",
                        metavar="MODEL_NAME")
    parser.add_argument("-lr", "--learning_rate",
                        help="Learning rate to run the optimizer function with.",
                        default=0.0001,
                        type=float,
                        metavar="FLOAT")
    parser.add_argument("--cuda",
                        help="Add this argument to run the code using GPU acceleration.",
                        action="store_true")
    parser.add_argument("-e", "--epochs",
                        help="Dictate number of epochs to train for.",
                        type=int,
                        metavar="INT",
                        default=5)

    args = parser.parse_args()

    device = "cpu"

    if args.cuda and torch.cuda.is_available(): 
        print("Using hardware acceleration")   
        device = "cuda"
    elif args.cuda:
        print("CUDA not available. Training on CPU instead")

    print(os.getcwd())

    train_dataset = CustomImageDataset('../images/train', annotation_path="../images/train_annotations", transforms=None)
    train_dataloader = DataLoader(train_dataset, 
                                    batch_size=args.batch, 
                                    shuffle=True)
    
    valid_dataset = CustomImageDataset('../images/valid', annotation_path="../images/valid_annotations", transforms=None)
    valid_dataloader = DataLoader(valid_dataset, 
                                    batch_size=args.batch, 
                                    shuffle=True)

    print(f"Training {args.model} with batch_size={args.batch}\n")

    model = None
    if args.model == "resnet18":
        model = resnet18().to(device)
    elif args.model == "resnet34":
        model = resnet34().to(device)
    elif args.model == "resnet50":
        model = custom_resnet50().to(device)
    elif args.model == "detr":
        model = DETR().to(device)
    elif args.model == "face_detr":
        model = train_detr.run(args, device, train_dataset, valid_dataset)

    # Train model
    if args.model != "face_detr":
        model = train(model, train_dataloader, args.learning_rate, device, valid_dataloader, epochs=args.epochs)

    torch.save(model.state_dict(), "model.pth")