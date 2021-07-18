# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import os

# Defaults to where the code is contained.
# If your bokeh input data is placed elsewhere, go ahead and edit this path
ROOT = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.abspath(os.path.join(ROOT, '..'))

# Intermediate bokeh files are cached under this path
CACHE_DIR = os.path.join(ROOT, 'cache')

# Bokeh inputs will be loaded from the following path
INPUT_DIR = os.path.join(ROOT, 'datasets', 'doc_images')
USER_STUDY_ROOT = os.path.join(ROOT, 'user_study')

# Default font embedding location, used when not specified explicitly
FONT_EMBEDDING_MODEL = os.path.join(ROOT, 'models', 'resnet_font_classifier.pth')