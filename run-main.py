from azureml.core import Workspace, Experiment, ScriptRunConfig

if __name__ == "__main__":
    env = ws.environments['AzureML-tensorflow-2.4-ubuntu18.04-py37-cpu-inference']
    config = ScriptRunConfig(source_directory='./src',
                             script='main.py',
                             arguments=['--epochs', 2,
                                        '--batches', 8, '--size', 16],
                             environment=env)

    run = experiment.submit(config)
    run.wait_for_completion()

    aml_url = run.get_portal_url()
    print(aml_url)
