# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import os
from pathlib import Path
from multimodal_affinities import common_config

"""
Controls which images may load for bokeh server, and whether user study mode should be initiated.
For casual inference runs, there shouldn't be any need to change this file directly.
"""

def any_image_in_datafolder():
   return [str(Path(entry).stem) for entry in os.listdir(common_config.INPUT_DIR) if entry.endswith('.json')]


input_config = {
    'data_folder': common_config.INPUT_DIR,    # Images and jsons for bokeh should reside here
    'file_basename': any_image_in_datafolder(),  # Should contain list of all filenames bokeh can randomly sample to load
    'is_user_study': False,
    'user_study_output_dir': os.path.join(common_config.USER_STUDY_ROOT, 'user_study_results')
}
