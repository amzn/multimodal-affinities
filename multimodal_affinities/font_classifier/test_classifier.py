"""
    general_training_flow.py
"""

from __future__ import print_function
from __future__ import division

import os


import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import models_factory
import optimizer_factory
import loss_factory
import datasources_factory
import copy
import argparse
import logging

def visualize_array(y_list, legend_list, title, output_fn):
    plt.clf()
    for y in y_list:
        plt.plot(y, '-')
    plt.legend(legend_list)
    plt.title(title)
    plt.savefig(output_fn)

def test_model(model, dataloaders, output_details=None):

    since = time.time()

    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Each epoch has a training and validation phase
    phase = 'val'

    running_loss = 0.0
    running_corrects = 0
    running_corrects_topk = 0

    # Iterate over data.
    num_images = 0
    for inputs, labels in dataloaders[phase]:

        # inputs = inputs.to(device)
        # labels = labels.to(device)

        outputs = model(inputs)
        _, predictedTop5 = outputs.topk(5)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        running_corrects_topk += torch.sum(torch.FloatTensor([1 for i,predicted_arr in enumerate(predictedTop5) if labels.data[i] in predicted_arr]))

        num_images += len(labels)
        if num_images > 1000:
            break

    accuracy_1 = running_corrects.double() / num_images
    accuracy_5 = running_corrects_topk.double() / num_images

    print('Top 1-class accuracy: %f' % accuracy_1)
    print('Top 5-class accuracy: %f' % accuracy_5)



if __name__ == '__main__':

    plt.ion()
    np.set_printoptions(suppress=True)

    np.random.seed(0)
    torch.manual_seed(0)

    # read the config file name
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default = '/font_classifier/config_files/train_cfg.json')
    parser.add_argument('--model_iteration', default=-1)
    args = parser.parse_args()
    run_cfg_file = args.config

    # load run params
    with open(run_cfg_file, 'r') as config_file:
        run_params = json.load(config_file)

    output_details = run_params['output_details']
    if not os.path.isdir(output_details['out_dir']):
         os.makedirs(output_details['out_dir'])

    # load the trained network
    model_iteration = args.model_iteration
    if model_iteration == -1:
        # get latest model
        allfiles = [file for file in os.listdir(output_details['out_dir']) if file.startswith(output_details['out_pfx'] + 'checkpoint')]
        iteration_num  = [int(file.split('-')[-1].split('.')[0]) for file in allfiles]
        model_iteration = np.sort(iteration_num)[-1]

    model_file = os.path.join(output_details['out_dir'], output_details['out_pfx'] + 'checkpoint-model-' + str(model_iteration) + '.pth')
    print('Model for evaluation: %s' % model_file)
    our_net = torch.load(model_file)
    checkpoint = torch.load(model_file)
    model_details = run_params['model_details']
    our_model, input_size = models_factory.models_factory(model_details['model_name'], model_details['model_params'])
    our_model.load_state_dict(checkpoint)
    our_model.eval()


    if our_model is None:
        logging.error('no model returned ...')
        exit()

    # source for training and validation or testing data
    data_details = run_params['dataset_details']
    data_details['dataset_params']['data_type'] = ['val']
    data_provider = datasources_factory.datasources_factory(data_details['dataset_name'], data_details['dataset_params'])

    if data_provider is None:
        logging.error('no data provider returned...')
        exit()

    # print some data about the network:

    logging.info('Our network: ')    # this prints the modules that were defined in the constructor of the net
    logging.info(our_model)


    # Test and evaluate
    test_model(our_model, data_provider, output_details=run_params['output_details'])



