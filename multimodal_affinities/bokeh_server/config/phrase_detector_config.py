# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

from bokeh.models.widgets import Slider

phrase_detector_config = {
    "Graph Clustering": {
        'horizontal_threshold': {
            'type': Slider,
            'title': 'horizontal threshold',
            'value': 0.55,
            'start': 0.0,
            'end': 2.0,
            'step': 0.01
        },
        'vertical_threshold': {
            'type': Slider,
            'title': 'vertical threshold',
            'value': 0.0,
            'start': 0.0,
            'end': 1,
            'step': 0.02
        },
        'post_processing_ordinals_threshold': {
            'type': Slider,
            'title': 'post_processing_ordinals_threshold',
            'value': 2.0,
            'start': 0.0,
            'end': 3,
            'step': 0.05
        }
    },
    "DBScan": {
        'epsilon': {
            'type': Slider,
            'title': 'eps',
            'value': 0.25,
            'start': 0.0,
            'end': 1.0,
            'step': 0.01
        },
        'min_samples': {
            'type': Slider,
            'title': 'min_samples',
            'value': 2,
            'start': 1,
            'end': 30,
            'step': 1
        }
    }
}
