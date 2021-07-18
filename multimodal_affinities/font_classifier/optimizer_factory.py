# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

"""
  optimizer_factory.py

  returns various optimizers to be used. You need to specify the trainable params - the variables over which we optimize

"""

import torch.optim

def optimizer_factory(optimizer_name, optimizer_params, trainable_params):
    """
         creates one of a few optimizers. we return None if we are unable to create
         the requested optimizer
    """

    if optimizer_name == 'SGD':

        l_rate = optimizer_params['learning_rate']
        momentum_val = optimizer_params['momentum']
        our_optimizer = torch.optim.SGD(trainable_params, lr = l_rate, momentum = momentum_val)
        return our_optimizer

    if optimizer_name == 'adam':
        our_optimizer = torch.optim.Adam(trainable_params)
        return our_optimizer


    return None
