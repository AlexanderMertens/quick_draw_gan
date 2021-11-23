from mlflow.tracking.fluent import end_run, log_artifact, start_run

from models.train_model import train_model

start_run()

train_model(num_epochs=2, num_batch=4, batch_size=8)
log_artifact('./figures/results', artifact_path='figures')

end_run()
