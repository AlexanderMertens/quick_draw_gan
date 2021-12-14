from azureml.core import Model
from azureml.core.workspace import Workspace


def register_model(name: str, type: str, run):
    """
    Registers TensorFlow model located at outputs/final_<type>.
    Its model name is <name>.
    The model is registered under the given run.
    """
    path = 'outputs/{}'.format(type)

    run.register_model(model_name='{}-{}'.format(name, type),
                       model_path=path, model_framework='TensorFlow')


def get_model(name: str) -> Model:
    """Returns Azure Model corresponding to given name.

    Args:
        name (str): Name of the registered Azure Model.

    Returns:
        Model: Container of Azure Model.
    """
    ws = Workspace.from_config()
    model = Model(workspace=ws, name=name)
    return model


def download_model(name: str):
    """Downloads Azure Model to current working directory.

    Args:
        name (str): Name of the registered Azure Model.
    """
    model = get_model(name)
    model.download(target_dir="./saved_model", exist_ok=True)
