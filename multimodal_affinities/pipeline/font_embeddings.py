from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import json
from multimodal_affinities.font_classifier.models_factory import models_factory
from multimodal_affinities.font_classifier.config_files import inference_cfg


class FontEmbeddings:
    """
    A class for converting word / phrases crops into an embedding vector representing the font.
    """

    def __init__(self, config=None, trained_model_path=None, num_classes=118):
        """
        :param config: Path to json or None if should use the default .py config at inference_cfg.py
        :param trained_model_path: Override config with a different trained model path
        """

        self.num_images_in_batch = 128

        if config:
            with open(config, 'r') as config_file:
                model_config = json.load(config_file)
        else:
            model_config = inference_cfg.font_inference_config

        if trained_model_path:
            model_config['trained_model_file'] = trained_model_path

        model_config['model_details']['model_params']['num_classes'] = num_classes

        # load the trained network
        map_location = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(model_config['trained_model_file'], map_location=map_location)
        model_details = model_config['model_details']
        model, input_size = models_factory(model_details['model_name'], model_details['model_params'])
        model.load_state_dict(checkpoint)
        model.eval()

        model_full, input_size = models_factory(model_details['model_name'], model_details['model_params'])
        model_full.load_state_dict(checkpoint)
        model_full.eval()

        # remove last fully-connected layer
        if model_details['model_params']['pretrained_model_name'] == 'resnet':
            new_classifier = nn.Sequential(*list(model.fc.children())[:-1])  # resnet
            model.fc = new_classifier
        elif model_details['model_params']['pretrained_model_name'] == 'vgg':
            new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])  # vgg
            model.classifier = new_classifier

        # Define transformations for the image, should (note that imagenet models are trained with image size 224)
        self.transformation = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            # Converts a PIL.Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape
            # (C x H x W) in the range [0.0, 1.0].
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # Normalize an tensor image with mean and standard deviation.
        ])

        self.model = model
        self.model_full = model_full
        if torch.cuda.is_available():
            self.model.cuda()
            self.model_full.cuda()

    def forward(self, images):
        with torch.no_grad():
            output_embedding_all = None
            for first_ind in range(0,len(images),self.num_images_in_batch):
                last_ind = min(first_ind + self.num_images_in_batch,len(images))
                # Preprocess the images
                image_tensor = torch.stack([self.transformation(image).float() for image in images[first_ind:last_ind]])

                if torch.cuda.is_available():
                    image_tensor = image_tensor.cuda()

                # Turn the input into a Variable
                input = Variable(image_tensor)

                output_embedding = self.model(input)
                output_softmax = self.model_full(input)
                _, predicted = torch.max(output_softmax, 1)

                if output_embedding_all is None:
                    output_embedding_all = output_embedding
                else:
                    output_embedding_all = torch.cat((output_embedding_all, output_embedding))
            return output_embedding_all


    def forward_debug(self, images):
        with torch.no_grad():
            output_embedding_all = None
            predicted_all = None
            for first_ind in range(0,len(images),self.num_images_in_batch):
                last_ind = min(first_ind + self.num_images_in_batch,len(images))
                # Preprocess the images
                image_tensor = torch.stack([self.transformation(image).float() for image in images[first_ind:last_ind]])

                if torch.cuda.is_available():
                    image_tensor = image_tensor.cuda()

                # Turn the input into a Variable
                input = Variable(image_tensor)

                output_embedding = self.model(input)
                output_softmax = self.model_full(input)
                _, predicted = torch.max(output_softmax, 1)

                if output_embedding_all is None:
                    output_embedding_all = output_embedding
                    predicted_all = predicted
                else:
                    output_embedding_all = torch.cat((output_embedding_all, output_embedding))
                    predicted_all = torch.cat((predicted_all, predicted))
            return output_embedding_all, predicted_all