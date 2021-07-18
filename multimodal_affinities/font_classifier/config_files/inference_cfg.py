# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

from multimodal_affinities import common_config

font_inference_config = {
     "model_details": {
                   "model_name": "pretrained_classifier",
                   "model_params": {

                                  "dbg_flag": False,
                                  "num_classes": 19,
                                  "pretrained_model_name": "resnet",
                                  "use_pretrained": True,
                                  "update_only_last_layer": False
                                 }
     },
    "trained_model_file": common_config.FONT_EMBEDDING_MODEL,
    "test_fn": []
}