"""
Script to setup and run experiment in azure ML studio.
Saves and registers the final models.
"""
import src.data_help.data_constants as dc

from azureml.core import Workspace, Experiment, ScriptRunConfig
from src.utility.azure_help import register_model

if __name__ == "__main__":
    ws = Workspace.from_config()
    env = ws.environments['AzureML-tensorflow-2.4-ubuntu18.04-py37-cuda11-gpu']
    experiment = Experiment(workspace=ws, name='12-10-WGAN-training')

    # Configure parameters
    num_epochs = 1
    num_batch = 150
    batch_size = 1024
    model_name = 'WGAN-Dog'

    # Setup script to run experiment
    config = ScriptRunConfig(source_directory='.', script='./src/main.py',
                             arguments=['--path', dc.TMP_DOG_DATA_PATH,
                                        '--epochs', num_epochs,
                                        '--batches', num_batch,
                                        '--size', batch_size],
                             environment=env)

    run = experiment.submit(config)
    run.wait_for_completion(show_output=True)

    # register final models to azure ml
    register_model(model_name, 'wgan', run)

    aml_url = run.get_portal_url()
    print(aml_url)
