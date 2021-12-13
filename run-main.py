"""
Script to setup and run experiment in azure ML studio.
Saves and registers the final models.
"""
from azureml.core import Workspace, Experiment, ScriptRunConfig
from src.utility.azure_help import register_model
from src.utility.load_config import load_config

if __name__ == "__main__":
    ws = Workspace.from_config()
    env = ws.environments['AzureML-tensorflow-2.4-ubuntu18.04-py37-cuda11-gpu']
    experiment = Experiment(workspace=ws, name='13-13-experiment-train')

    # Configure parameters
    num_epochs = 80
    num_batch = 128
    batch_size = 1024
    data_path = load_config('config.yaml')['data']['folder']
    name = 'whale'
    model_name = 'DCGAN-{}'.format(name.capitalize())

    # Setup script to run experiment
    config = ScriptRunConfig(source_directory='.', script='./src/main.py',
                             arguments=['--path', data_path,
                                        '--name', name,
                                        '--epochs', num_epochs,
                                        '--batches', num_batch,
                                        '--size', batch_size],
                             environment=env)

    run = experiment.submit(config)
    run.wait_for_completion(show_output=True)

    # register final models to azure ml
    register_model(model_name, 'discriminator', run)
    register_model(model_name, 'generator', run)

    aml_url = run.get_portal_url()
    print(aml_url)
