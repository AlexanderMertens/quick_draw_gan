"""
This script downloads dataset to a temporary folder.
"""
# azureml-core of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset

import os
if not os.path.exists('/tmp/quick-draw-data'):
    os.makedirs('/tmp/quick-draw-data')

# get workspace
subscription_id = 'a8260178-3b6d-4bce-a07e-3aae8c7a62af'
resource_group = 'RG_GAN_internship'
workspace_name = 'Quick_Draw_Project'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='quick_draw')
dataset.download(target_path='/tmp/quick-draw-data/', overwrite=False)
