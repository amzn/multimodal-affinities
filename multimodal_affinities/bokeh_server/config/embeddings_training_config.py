# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

from bokeh.models.widgets import CheckboxGroup, Select, TextInput, Slider, MultiSelect

embeddings_training_config = {

    "embedding_name": 'Embeddings Training Config',
    "is_editable": True,
    "parameters": {
        'ae_dims_1': {
            'type': TextInput,
            'value': '',
            'title': 'Autoenc. Layer 1',
        },
        'ae_dims_2': {
            'type': TextInput,
            'value': '500',
            'title': 'Autoenc. Layer 2',
        },
        'ae_dims_3': {
            'type': TextInput,
            'value': '2000',
            'title': 'Autoenc. Layer 3',
        },
        'ae_dims_4': {
            'type': TextInput,
            'value': '20',
            'title': 'Autoenc. Layer 4',
        },
        'dropout': {
            'type': Slider,
            'title': 'dropout',
            'value': 0.2,
            'start': 0.0,
            'end': 1.0,
            'step': 0.01
        },
        'batch_size': {
            'type': Select,
            'title': 'batch_size',
            'options': ['1', '4', '8', '16', '32', '64', '128', '256', '512', '1024'],
            'value': '32'
        },
        'learning_rate': {
            'type': Select,
            'title': 'learning_rate',
            'options': ['1e-2', '1e-3', '1e-4', '1e-5', '1e-6'],
            'value': '1e-4',
        },
        'epochs': {
            'type': Slider,
            'title': 'epochs',
            'value': 60,
            'start': 0,
            'end': 200,
            'step': 1
        },
        'early_stop': {
            'type': CheckboxGroup,
            'labels': ['early_stop'],
            'active': []
        },
        'gradient_clip': {
            'type': Slider,
            'title': 'gradient_clip',
            'value': 5,
            'start': 0,
            'end': 10,
            'step': 0.1
        },
        'push_pull_weight_ratio': {
            'type': Slider,
            'title': 'push_pull_weight_ratio',
            'value': 3.0,
            'start': 0.5,
            'end': 10,
            'step': 0.01
        },
        'push_pull_decay': {
            'type': Slider,
            'title': 'push_pull_decay',
            'value': 0.86,
            'start': 0.6,
            'end': 1.0,
            'step': 0.01
        },
        'reconstruction_loss_normalization': {
            'type': Slider,
            'title': 'reconstruction_loss_normalization',
            'value': 0.0,
            'start': 0.0,
            'end': 2.0,
            'step': 0.01
        },
        'normalize_embeddings_pre_process': {
            'type': CheckboxGroup,
            'labels': ['normalize_embeddings_pre_process'],
            'active': []
        },
        'normalize_embeddings_post_process': {
            'type': CheckboxGroup,
            'labels': ['normalize_embeddings_post_process'],
            'active': []
        },
        'downsample_user_constraints': {
            'type': CheckboxGroup,
            'labels': ['downsample_user_constraints'],
            'active': [0]
        },
        'downsample_default_constraints': {
            'type': CheckboxGroup,
            'labels': ['downsample_default_constraints'],
            'active': [0]
        },
        'must_cannot_ratio': {
            'type': TextInput,
            'value': '1.5',
            'title': 'must_cannot_ratio',
        },
        'font_word_mknn': {
            'type': TextInput,
            'value': '6',
            'title': 'font_word_mknn',
        },
        'max_mustlink': {
            'type': TextInput,
            'value': '1000',
            'title': 'max_mustlink',
        },
        'constraints_types': {
            'type': MultiSelect,
            'title': 'constraints_types',
            'options': [
                ("font", "font"),
                ("nlp-ner", "nlp-ner"),
                ("geometry", "geometry")
            ],
            'value': ["font", "nlp-ner", "geometry"]
        }
    }
}
