from azureml.core import Model


def register_model(name, type, run):
    path = 'outputs/final_{}'.format(type)
    output_directory = './outputs/{}'.format(type)

    run.download_files(
        prefix=path, output_directory=output_directory, append_prefix=False)
    run.register_model(model_name='{}-{}'.format(name, type),
                       model_path=output_directory, model_framework='TensorFlow')
