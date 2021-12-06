from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
import tensorflow as tf

from models.train_model import train_model


subscription_id = 'a8260178-3b6d-4bce-a07e-3aae8c7a62af'
resource_group = 'RG_GAN_internship'
workspace_name = 'Quick_Draw_Project'

ws = Workspace(subscription_id, resource_group, workspace_name)

experiment = Experiment(workspace=ws, name='12-06-experiment-train')
run = experiment.start_logging(
    outputs=None, snapshot_directory='.', display_name='First trial')

gan = train_model(run=run, num_epochs=2, num_batch=4,
                  batch_size=16)
run.complete()
