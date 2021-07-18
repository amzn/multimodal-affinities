# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

"""
  loss_factory.py

  returns various net models to be used
"""

import torch.nn

def loss_factory(loss_name, loss_params = None):
     """
         returns a loss function to be used in training. we return None if we are unable to create
         the requested loss
     """

     if loss_name == 'NLL':

        loss_function = torch.nn.NLLLoss()

        return loss_function

     if loss_name == 'cross_entropy':

         loss_function = torch.nn.CrossEntropyLoss()

         return loss_function


     return None
