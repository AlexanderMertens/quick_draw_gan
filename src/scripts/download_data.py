"""
This script downloads dataset to a temporary folder.
"""
# azureml-core of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset

import os


def download_data(data_path: str, name: str):
    """Download data to temporary folder.
    If data is already downloaded, cancel the operation.

    Args:
        data_path (str): path of the required data
        name (str): Name of the dataset that will be used
    """
    # Check if data is already downloaded, if so cancel operation
    if os.path.exists('{}/{}.npy'.format(data_path, name)):
        return

    # Create map to download data to
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    workspace = Workspace.from_config()

    dataset = Dataset.get_by_name(workspace, name='quick_draw')
    dataset.download(target_path=data_path, overwrite=True)
