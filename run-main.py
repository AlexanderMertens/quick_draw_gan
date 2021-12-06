from azureml.core import Workspace, Experiment, ScriptRunConfig

if __name__ == "__main__":
    ws = Workspace.from_config()
    env = ws.environments['AzureML-tensorflow-2.4-ubuntu18.04-py37-cpu-inference']
    experiment = Experiment(workspace=ws, name='12-06-experiment-train')

    num_epochs = 2
    num_batch = 2
    batch_size = 16

    config = ScriptRunConfig(source_directory='.',
                             script='./src/main.py',
                             arguments=['--epochs', 2,
                                        '--batches', 2, '--size', 16],
                             environment=env)

    run = experiment.submit(config)
    run.wait_for_completion()

    aml_url = run.get_portal_url()
    print(aml_url)
