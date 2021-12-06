from azureml.core import Workspace, Experiment, ScriptRunConfig, Model

from src.utility.azure_help import register_model

if __name__ == "__main__":
    ws = Workspace.from_config()
    env = ws.environments['AzureML-tensorflow-2.4-ubuntu18.04-py37-cpu-inference']
    experiment = Experiment(workspace=ws, name='12-06-experiment-train')

    num_epochs = 1
    num_batch = 2
    batch_size = 16
    model_name = 'DCGAN-butterfly'

    config = ScriptRunConfig(source_directory='.',
                             script='./src/main.py',
                             arguments=['--epochs', num_epochs, '--batches', num_batch,
                                        '--size', batch_size],
                             environment=env)

    run = experiment.submit(config)
    run.wait_for_completion(show_output=True)

    # register final models to azure ml
    register_model(model_name, 'discriminator', run)
    register_model(model_name, 'generator', run)

    aml_url = run.get_portal_url()
    print(aml_url)
