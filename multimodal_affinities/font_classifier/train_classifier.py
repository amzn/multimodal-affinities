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

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, num_epochs_to_save=10, output_details=None):

    since = time.time()

    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            bc = 0
            for inputs, labels in dataloaders[phase]:
                if bc % 100 == 0:
                    print('Epoch {}: batch {}'.format(epoch, bc))
                bc += 1

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # if bc % 100 == 0:
                #     output_fn = os.path.join(output_details['out_dir'],
                #                              output_details['out_pfx'] + '-accuracy-' + 'epoch-' + str(epoch)+ '-batch-' + str(bc) + '.png')
                #     visualize_array([train_acc_history, val_acc_history], ['train', 'val'], 'Accuracy', output_fn)



            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logging.info('Epoch {} - {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            else:
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

            if epoch % num_epochs_to_save == 0:
                # save model
                torch.save(best_model_wts, os.path.join(output_details['out_dir'],
                                                        output_details['out_pfx'] + 'checkpoint-model-' + str(
                                                            epoch) + '.pth'))
                logging.info('saved checkpoint model for epoch %d and batch counter %d ... \n' % (epoch, 0))

                output_fn = os.path.join(output_details['out_dir'],
                                         output_details['out_pfx'] + 'accuracy.png')
                visualize_array([train_acc_history, val_acc_history], ['train', 'val'], 'Accuracy', output_fn)
                output_fn = os.path.join(output_details['out_dir'],
                                         output_details['out_pfx'] + 'loss..png')
                visualize_array([train_loss_history, val_loss_history], ['train', 'val'], 'Loss', output_fn)




    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, val_loss_history, train_loss_history



if __name__ == '__main__':

    plt.ion()
    np.set_printoptions(suppress=True)

    np.random.seed(0)
    torch.manual_seed(0)

    # read the config file name
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/font_classifier/config_files/train_cfg.json')
    args = parser.parse_args()
    run_cfg_file = args.config

    # load run params
    with open(run_cfg_file, 'r') as config_file:
        run_params = json.load(config_file)

    output_details = run_params['output_details']
    if not os.path.isdir( output_details['out_dir']):
         os.makedirs(output_details['out_dir'])

    logger_file = os.path.join(output_details['out_dir'], output_details['out_pfx'] + 'log.txt')
    logging.basicConfig(filename=logger_file, level=logging.DEBUG)

    # define our NN
    model_details = run_params['model_details']
    feature_extract = model_details['model_params']['update_only_last_layer']
    our_model, input_size = models_factory.models_factory(model_details['model_name'], model_details['model_params'])

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    our_model = our_model.to(device)

    if our_model is None:
        logging.error('no model returned ...')
        exit()

    # source for training and validation or testing data
    data_details = run_params['dataset_details']
    data_provider = datasources_factory.datasources_factory(data_details['dataset_name'], data_details['dataset_params'])

    if data_provider is None:
        logging.error('no data provider returned...')
        exit()

    # print some data about the network:

    logging.info('Our network: ')    # this prints the modules that were defined in the constructor of the net
    logging.info(our_model)


    # get a loss function

    loss_details = run_params['loss_details']
    our_loss_function = loss_factory.loss_factory(loss_details['loss_name'], loss_details['loss_params'])

    if our_loss_function is None:
        logging.error('no loss function returned ...')
        exit()

    #
    #  Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    #
    params_to_update = our_model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in our_model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in our_model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    #
    # get an optimizer
    #

    opt_details = run_params['optimizer_details']
    our_optimizer = optimizer_factory.optimizer_factory(opt_details['optimizer_name'], opt_details['optimizer_params'], params_to_update)

    if our_optimizer is None:
        logging.error('no optimizer returned ...')
        exit()

    # Train and evaluate
    our_model, val_acc, train_acc, val_loss, train_loss = train_model(our_model, data_provider, our_loss_function, our_optimizer, num_epochs=run_params['training_details']['epochs'],
                                                                      is_inception=(run_params['model_details']['model_name'] == "inception"), num_epochs_to_save=5, output_details=run_params['output_details'])

    # save final model and visualize accuracy + loss
    torch.save(our_model, os.path.join(output_details['out_dir'],
                                            output_details['out_pfx'] + 'checkpoint-model-final' + '.pth'))

    output_fn = os.path.join(output_details['out_dir'],
                             output_details['out_pfx'] + 'accuracy-final.png')
    visualize_array([train_acc, val_acc], ['train', 'val'], 'Accuracy', output_fn)
    output_fn = os.path.join(output_details['out_dir'],
                             output_details['out_pfx'] + 'loss-final.png')
    visualize_array([train_loss, val_loss], ['train', 'val'], 'Loss', output_fn)


