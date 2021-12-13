"""
This script downloads dataset to a temporary folder.
"""
# azureml-core of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset

import os
if not os.path.exists('/tmp/quick-draw-data'):
    os.makedirs('/tmp/quick-draw-data')

workspace = Workspace.from_config()

dataset = Dataset.get_by_name(workspace, name='quick_draw')
dataset.download(target_path='/tmp/quick-draw-data/', overwrite=False)
