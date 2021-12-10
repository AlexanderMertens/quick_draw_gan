from azureml.core import Model


def register_model(name: str, type: str, run):
    """
    Registers TensorFlow model located at outputs/final_<type>.
    Its model name is <name>.
    The model is registered under the given run.
    """
    path = 'outputs/final_{}'.format(type)

    print("saving model at", path)
    run.register_model(model_name='{}-{}'.format(name, type),
                       model_path=path, model_framework='TensorFlow')
