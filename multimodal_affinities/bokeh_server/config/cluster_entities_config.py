# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

from bokeh.models.widgets import Slider, Select

cluster_entities_config = {
    "Graph Clustering": {
        'threshold': {
            'type': Slider,
            'title': 'threshold',
            'value': 0.15,
            'start': 0.0,
            'end': 1.0,
            'step': 0.025
        },
        'mst_threshold': {
            'type': Slider,
            'title': 'mst_threshold',
            'value': 0.5,
            'start': 0.0,
            'end': 1.0,
            'step': 0.025
        },
        'max_island_cluster_size': {
            'type': Slider,
            'title': 'max_island_cluster_size',
            'value': 1,
            'start': 1,
            'end': 10,
            'step': 1
        },
        'ratio_to_mean_dist_to_isolate': {
            'type': Slider,
            'title': 'ratio_to_mean_dist_to_isolate',
            'value': 2.0,
            'start': 0,
            'end': 10,
            'step': 0.1
        },
        'distance_metric': {
            'type': Select,
            'title': 'distance_metric',
            'options': ["neural_affinity", "embedding_distance"],
            'value': "neural_affinity"
        },
        'height_ratio_cutoff_threshold': {
            'type': Slider,
            'title': 'height_ratio_cutoff_threshold',
            'value': 0.25,
            'start': 0.0,
            'end': 1.0,
            'step': 0.025
        },
        'reduce': {
            'type': Select,
            'title': 'reduce',
            'options': ["mean", "mean_without_log", "median", "median_no_log"],
            'value': "mean_without_log"
        },
    },
    "DBScan": {
        'epsilon': {
            'type': Slider,
            'title': 'eps',
            'value': 0.25,
            'start': 0.0,
            'end': 5.0,
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
