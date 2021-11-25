from mlflow.tracking.fluent import end_run, log_artifact, start_run
from mlflow.keras import log_model

from models.train_model import train_model

start_run()

gan = train_model(num_epochs=1, num_batch=4, batch_size=8)
log_artifact('./figures/results', artifact_path='figures')
log_model(gan, 'my_model', conda_env='./conda.yaml')

end_run()
