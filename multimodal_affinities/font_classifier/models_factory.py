"""
  models_factory.py

  returns various net models to be used
"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def models_factory(model_name, model_params):
     """
         creates one of a few implementations of a network. we return None if we are unable to create
         the requested network
     """
     if model_name == 'pretrained_classifier':

         model_name = model_params['pretrained_model_name']
         num_classes = model_params['num_classes']
         use_pretrained = model_params['use_pretrained']
         # If feature_extract = False, the model is finetuned and all model parameters are updated.
         # If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
         feature_extract = model_params['update_only_last_layer']
         # Initialize these variables which will be set in this if statement. Each of these
         #   variables is model specific.
         model_ft = None
         input_size = 0

         if model_name == "resnet":
             """ Resnet18
             """
             model_ft = models.resnet18(pretrained=use_pretrained)
             set_parameter_requires_grad(model_ft, feature_extract)
             num_ftrs = model_ft.fc.in_features
             model_ft.fc = nn.Linear(num_ftrs, num_classes)
             input_size = 224

         elif model_name == "alexnet":
             """ Alexnet
             """
             model_ft = models.alexnet(pretrained=use_pretrained)
             set_parameter_requires_grad(model_ft, feature_extract)
             num_ftrs = model_ft.classifier[6].in_features
             model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
             input_size = 224

         elif model_name == "vgg":
             """ VGG11_bn
             """
             model_ft = models.vgg11_bn(pretrained=use_pretrained)
             set_parameter_requires_grad(model_ft, feature_extract)
             num_ftrs = model_ft.classifier[6].in_features
             model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
             input_size = 224

         elif model_name == "squeezenet":
             """ Squeezenet
             """
             model_ft = models.squeezenet1_0(pretrained=use_pretrained)
             set_parameter_requires_grad(model_ft, feature_extract)
             model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
             model_ft.num_classes = num_classes
             input_size = 224

         elif model_name == "densenet":
             """ Densenet
             """
             model_ft = models.densenet121(pretrained=use_pretrained)
             set_parameter_requires_grad(model_ft, feature_extract)
             num_ftrs = model_ft.classifier.in_features
             model_ft.classifier = nn.Linear(num_ftrs, num_classes)
             input_size = 224

         elif model_name == "inception":
             """ Inception v3 
             Be careful, expects (299,299) sized images and has auxiliary output
             """
             model_ft = models.inception_v3(pretrained=use_pretrained)
             set_parameter_requires_grad(model_ft, feature_extract)
             # Handle the auxilary net
             num_ftrs = model_ft.AuxLogits.fc.in_features
             model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
             # Handle the primary net
             num_ftrs = model_ft.fc.in_features
             model_ft.fc = nn.Linear(num_ftrs, num_classes)
             input_size = 299

         else:
             print("Invalid model name, exiting...")
             exit()

         return model_ft, input_size

     return None
