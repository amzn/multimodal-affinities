# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

"""
  datasources_factory.py

  returns various data sources to be used
"""

import torch
from torchvision import datasets, transforms
import os


def datasources_factory(dataset_name, dataset_params):
     """
         creates one of a few implementations of a data source. we return None if we are unable to create
         the requested data source
     """


     if dataset_name == 'font-1':

         data_transforms = {
             'train': transforms.Compose([
                 transforms.Resize(size=(224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
             ]),
             'val': transforms.Compose([
                 transforms.Resize(size=(224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
             ]),
         }

         print("Initializing Datasets and Dataloaders...")

         data_type = dataset_params.get('data_type', ['train','val'])
         # Create training and validation datasets
         image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_params['root_dir'], x), data_transforms[x]) for x in data_type}
         # Create training and validation dataloaders
         dataloaders_dict = {
         x: torch.utils.data.DataLoader(image_datasets[x], batch_size=dataset_params['batch_size'], shuffle=True, num_workers=4) for x in data_type}

         return dataloaders_dict


     return None
